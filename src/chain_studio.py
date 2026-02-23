#!/usr/bin/env python3
"""
LangChain-style Chain Builder and Runner
Build and execute LLM chains without requiring LangChain dependency
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import requests


DB_PATH = Path.home() / ".blackroad" / "chains.db"
OLLAMA_API = "http://localhost:11434"


@dataclass
class ChainStep:
    """Represents a step in a chain"""
    id: str
    type: str  # llm, transform, condition, output
    name: str
    prompt_template: str
    model: str
    temperature: float
    max_tokens: int
    output_key: str


@dataclass
class Chain:
    """Represents a complete chain"""
    id: str
    name: str
    description: str
    steps: List[ChainStep] = field(default_factory=list)
    created_at: str = ""
    last_run: Optional[str] = None


class ChainStudio:
    """LLM Chain Builder and Runner"""
    
    def __init__(self):
        """Initialize studio with database"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chains (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT,
                last_run TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                chain_id TEXT NOT NULL,
                type TEXT,
                name TEXT,
                prompt_template TEXT,
                model TEXT,
                temperature REAL,
                max_tokens INTEGER,
                output_key TEXT,
                FOREIGN KEY(chain_id) REFERENCES chains(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_chain(self, name: str, description: str = "") -> Chain:
        """Create a new chain"""
        chain_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO chains (id, name, description, created_at) VALUES (?, ?, ?, ?)',
            (chain_id, name, description, now)
        )
        conn.commit()
        conn.close()
        
        return Chain(id=chain_id, name=name, description=description, created_at=now)
    
    def add_step(self, chain_id: str, type: str, name: str, 
                 prompt_template: str, model: str = "qwen2.5", 
                 temperature: float = 0.7) -> ChainStep:
        """Add step to chain"""
        step_id = str(uuid.uuid4())[:8]
        max_tokens = 2048
        output_key = name.lower().replace(" ", "_")
        
        step = ChainStep(
            id=step_id, type=type, name=name, 
            prompt_template=prompt_template, model=model,
            temperature=temperature, max_tokens=max_tokens, output_key=output_key
        )
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO steps (id, chain_id, type, name, prompt_template, model, 
                             temperature, max_tokens, output_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (step.id, chain_id, step.type, step.name, step.prompt_template,
              step.model, step.temperature, step.max_tokens, step.output_key))
        conn.commit()
        conn.close()
        
        return step
    
    def run_chain(self, chain_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chain steps sequentially"""
        chain = self.get_chain(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        context = dict(inputs)
        
        for step in chain.steps:
            if step.type == "llm":
                # Call Ollama API
                prompt = self._render_template(step.prompt_template, context)
                response = self._call_ollama(prompt, step.model, step.temperature, step.max_tokens)
                context[step.output_key] = response
            elif step.type == "transform":
                # Simple transformation
                context[step.output_key] = self._transform(step.prompt_template, context)
            elif step.type == "condition":
                # Conditional logic
                if not self._evaluate_condition(step.prompt_template, context):
                    break
        
        # Update last run
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE chains SET last_run = ? WHERE id = ?',
                      (datetime.now().isoformat(), chain_id))
        conn.commit()
        conn.close()
        
        return context
    
    def test_chain(self, chain_id: str, test_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Dry run with mock LLM"""
        chain = self.get_chain(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        context = dict(test_inputs)
        
        for step in chain.steps:
            if step.type == "llm":
                prompt = self._render_template(step.prompt_template, context)
                context[step.output_key] = f"[MOCK] Response to: {prompt[:50]}..."
            elif step.type == "transform":
                context[step.output_key] = self._transform(step.prompt_template, context)
        
        return context
    
    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """Get chain with step details"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, description, created_at, last_run FROM chains WHERE id = ?',
                      (chain_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Fetch steps
        cursor.execute('SELECT id, type, name, prompt_template, model, temperature, max_tokens, output_key '
                      'FROM steps WHERE chain_id = ? ORDER BY rowid', (chain_id,))
        steps_rows = cursor.fetchall()
        conn.close()
        
        steps = [
            ChainStep(id=s[0], type=s[1], name=s[2], prompt_template=s[3],
                     model=s[4], temperature=s[5], max_tokens=s[6], output_key=s[7])
            for s in steps_rows
        ]
        
        return Chain(id=row[0], name=row[1], description=row[2], 
                    steps=steps, created_at=row[3], last_run=row[4])
    
    def list_chains(self) -> List[Dict[str, Any]]:
        """List all chains with last run status"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, created_at, last_run FROM chains ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {'id': row[0], 'name': row[1], 'description': row[2], 
             'created_at': row[3], 'last_run': row[4]}
            for row in rows
        ]
    
    def export_chain(self, chain_id: str) -> Dict[str, Any]:
        """Export chain as portable JSON"""
        chain = self.get_chain(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        return {
            'id': chain.id,
            'name': chain.name,
            'description': chain.description,
            'created_at': chain.created_at,
            'steps': [asdict(s) for s in chain.steps]
        }
    
    def import_chain(self, json_path: str) -> Chain:
        """Import chain from JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        chain = self.create_chain(data['name'], data.get('description', ''))
        
        for step_data in data.get('steps', []):
            self.add_step(
                chain.id,
                step_data['type'],
                step_data['name'],
                step_data['prompt_template'],
                step_data.get('model', 'qwen2.5'),
                step_data.get('temperature', 0.7)
            )
        
        return chain
    
    def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Render prompt template with variables"""
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
    
    def _call_ollama(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{OLLAMA_API}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=30
            )
            return response.json().get('response', '')
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def _transform(self, template: str, context: Dict[str, Any]) -> str:
        """Simple transformation"""
        return self._render_template(template, context)
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate condition"""
        try:
            return bool(eval(condition, {"__builtins__": {}}, context))
        except:
            return False


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="LangChain Studio")
    subparsers = parser.add_subparsers(dest='command')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List chains')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create chain')
    create_parser.add_argument('name', help='Chain name')
    create_parser.add_argument('--desc', help='Description')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run chain')
    run_parser.add_argument('chain_id', help='Chain ID')
    run_parser.add_argument('--input', help='Input as key=value')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export chain')
    export_parser.add_argument('chain_id', help='Chain ID')
    export_parser.add_argument('--output', help='Output file')
    
    args = parser.parse_args()
    studio = ChainStudio()
    
    if args.command == 'list':
        chains = studio.list_chains()
        for chain in chains:
            print(f"{chain['id']} - {chain['name']} (Created: {chain['created_at']})")
    elif args.command == 'create':
        chain = studio.create_chain(args.name, args.desc or "")
        print(f"Created chain: {chain.id} ({chain.name})")
    elif args.command == 'run':
        inputs = {}
        if args.input:
            key, value = args.input.split('=')
            inputs[key] = value
        result = studio.run_chain(args.chain_id, inputs)
        print(json.dumps(result, indent=2))
    elif args.command == 'export':
        chain_data = studio.export_chain(args.chain_id)
        output = args.output or f"{args.chain_id}.json"
        with open(output, 'w') as f:
            json.dump(chain_data, f, indent=2)
        print(f"Exported to {output}")


if __name__ == '__main__':
    main()
