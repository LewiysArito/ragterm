from typing import Any, Dict, List, Optional, Union
from domain import DocumentVector
import re

document_vector = DocumentVector()

class CommandHandler:
    """
    Command handler
    """

    def __init__(self):
        self.commands = {
            "exit": {
                "description": "Exit the application",
                "function": self._exit_command,
                "args" : None
            },
            "parameters" : {
                "description": "Shows the parameters and description of each of them",
                "function": self._parameters_command,
                "args" : {"command": "Name of command to show parameters for "} 
            },
            "help": {
                "description": "Show help message", 
                "function": self._help_command,
                "args": None
            },
            "echo": {
                "description": "Echo back your message",
                "function": self._echo_command,
                "args": None
            },
            "upload" : {
                "description": "Upload file and create collections assotiated with it",
                "function": self._upload_command,
                "args" : {"filepath": "Path to file for upload"}
            },
            "delete" : {
                "description": "Delete collections associated with file",
                "function": self._delete_command,
                "args": {"filename": "Name of file to delete from vector database"}
            },
            "clear" : {
                "description": "Clear all collections",
                "function": self._clear_command,
                "args": None
            },
            "collections" : {
                "description": "Get list collections embedding db, uploaded to rag",
                "function": self._collections_command,
                "args": None
            },
            "chunks" : {
                "description": "Get all relevants chunk",
                "function": self._chunks_command,
                "args": {"filename": "Search source file", "query": "Search query"}
            },
            "result" : {
                "description": "Get RAG Result",
                "function": self._result_command,
                "args": {"filename": "Search source file", "query": "Search query"}
            }
        }
    
    def _commands_items(self)->List[str]:
        return list(self.commands.keys())

    def _exit_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle exit command"""
        print("User requested exit")
        return False
    
    def _help_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle help command"""
        print("\nAvailable commands:")
        for cmd, info in self.commands.items():
            print(f"  {cmd} - {info['description']}")
        print()
        return True

    def _clear_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        return True

    def _echo_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle echo command"""
        if args:
            print(" ".join(args))
        else:
            print("Usage: echo <message>")
        return True
    
    def _upload_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle upload command"""        

        if isinstance(args, dict) and args.get("filename"):
            document_vector.upload_file(args["filename"])
            print(f"File {args['filename']} uploaded successfully")
        elif isinstance(args, list) and args and args[0]:
            document_vector.upload_file(args[0])
            print(f"File {args[0]} uploaded successfully")
        else:
            print("No valid filename provided")
        
        return True
    
    def _delete_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        return True

    def _collections_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        return document_vector.show_all_collections()

    def _chunks_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        return True

    def _result_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        return True

    def execute_command(self, command_name, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Execute a command"""
        if command_name in self.commands:
            return self.commands[command_name]["function"](args)
        else:
            print(f"Unknown command: {command_name}. Type 'help' for available commands.")
            return True

def start_cli():
    handler = CommandHandler()
    is_active = True

    minimal = """
┌────────────────────────────────────────┐
│                                        │
│               RAGTERM                  │
│          Terminal Application          │
│    Retrieval-Augmented Generation      │
│                                        │
├────────────────────────────────────────┤
│                                        │
│  Version: 1.0.0                        │
│  Type 'help' for commands              │
│                                        │
└────────────────────────────────────────┘
"""
    print(minimal)
    commands_format = [
        "\n",
        "Commands Formats\n",
        "Option 1: Positional Arguments (without parameter names)",
        "> result 'cook book.pdf' 'How to make ratatouille?'",
        "> result filename.pdf query_without_spaces  # quotes optional for values without spaces\n",
        "Option 2: Named Arguments (with parameter names)",
        '> result --filename="cook book.pdf" --query="How to make ratatouille?"',
        "> result --filename=file.pdf --query=simple_query  # quotes optional for values without spaces\n"
    ]
    print("\n".join(commands_format))

    while is_active:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            parts = user_input.split()
            command_name = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []

            is_active = handler.execute_command(command_name, args)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Goodbye!")
            is_active = False
        except Exception as e:
            print(f"CLI error: {e}")
            print(f"Error: {e}")
            is_active = False
    
    print("CLI session completed")

if __name__ == "__main__":
    start_cli()
