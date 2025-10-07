from typing import Any, Dict, List
from domain import DocumentVector

document_vector = DocumentVector()

class CommandHandler:
    """
    Command handler
    """

    def __init__(self):
        self.commands = {
            "exit": {
                "description": "Exit the application",
                "function": self._exit_command
            },
            "help": {
                "description": "Show help message", 
                "function": self._help_command
            },
            "echo": {
                "description": "Echo back your message",
                "function": self._echo_command
            },
            "upload" : {
                "description": "Upload file and create collections assotiated with him",
                "function": self._upload_command
            },
            "delete" : {
                "description": "Delete collections associated with file",
                "function": self._delete_command
            }, 
            "collections" : {
                "description": "Get lish collections embedding db",
                "function": self._collections_command
            },
            "chunks" : {
                "description": "Get all relevants chunk",
                "function": self._chunks_command
            },
            "result" : {
                "description": "Get RAG Result",
                "function": self._result_command
            }
        }
    
    def _exit_command(self, args):
        """Handle exit command"""
        print("User requested exit")
        return False
    
    def _help_command(self, args):
        """Handle help command"""
        print("\nAvailable commands:")
        for cmd, info in self.commands.items():
            print(f"  {cmd} - {info['description']}")
        print()
        return True
    
    def _echo_command(self, args):
        """Handle echo command"""
        if args:
            print(" ".join(args))
        else:
            print("Usage: echo <message>")
        return True
    
    def _upload_command(self, args):
        """Handle upload command"""
        return True
        
    def _delete_command(self, args):
        return True

    def _collections_command(self, args):
        return document_vector.show_all_collections()

    def _chunks_command(self, args):
        return True

    def _result_command(self, args):
        return True

    def execute_command(self, command_name, args: List[str] | Dict[str, Any]):
        """Execute a command"""
        if command_name in self.commands:
            return self.commands[command_name]["function"](args)
        else:
            print(f"Unknown command: {command_name}. Type 'help' for available commands.")
            return True

def start_cli():
    handler = CommandHandler()
    is_active = True

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
