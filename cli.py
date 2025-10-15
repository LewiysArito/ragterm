from typing import Any, Dict, List, Optional, Union
from domain import DocumentVector
import re

document_vector = DocumentVector()

class CliArgumentsError(Exception):
    pass

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
                "args" : {"command": "Name of command to show parameters"} 
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
                "args" : {"file_path": "Path to file for upload"}
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

    def parser_args_positional(self, text: str) -> List[str]:
        pattern = r"['\"]([^'\"]+)['\"]|((?!--)[\S]+)"
        matches = re.findall(pattern, text)
        result = []

        for match in matches:
            if match[0]:
                result.append(match[0])
            elif match[1]:
                result.append(match[1])

        if not sorted(" ".join(result)) == sorted(text.replace('"', '').replace("'", "")):
            return None

        return result
    
    def parser_args_named(self, text: str) -> Optional[Dict[str, str]]:
        pattern = r'--([a-zA-Z_][a-zA-Z0-9_]*)=([^\"\'\s]+|[\"]([^\"]*)[\"]|[\']([^\']*)[\'])'
        matches = re.findall(pattern, text)
        result = {}

        for match in matches:
            key = match[0]
            if match[2]:
                value = match[2]
            elif match[3]:
                value = match[3]
            else:
                value = match[1]
            
            result[key] = value
        
        reconstructed_parts = []
        for key, value in result.items():
            reconstructed_parts.append(f"--{key}={value}")
        
        if not sorted(text) == sorted(" ".join(reconstructed_parts)):
            return None
        
        return result

    def parser_args(self, text:str):
        args = self.parser_args_named(text) or self.parser_args_positional(text) 
                
        if not args:
            raise CliArgumentsError("Arguments not is named or positional")

        return args
            
    def _commands_items(self)->List[str]:
        return list(self.commands.keys())
    
    def _parameters_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle parameters"""
        
        command = None
        if isinstance(args, dict) and args.get("command"):
            command = args.get("command")
        elif isinstance(args, list) and args and args[0]:
            command = args[0]
        else:
            print("No provided command parameter")
            return True

        item_command = self.commands.get(command)
        if not item_command:
            print(f"Command with name='{command}' not exists")
            return True
        
        parameters = item_command["args"]

        if not parameters:
            print(f"'{command}' has not parameters")    
            return True

        print(f"'{command}' parameters\n")
        for key, value in item_command["args"].items():
            print(f"--{key} - {value}")
        
        return True
    
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
        
        collections = document_vector.clear_all()
        if not collections:
            print("Collections is empty")
            return True

        print("Deleted collections:")
        for collection in collection:
            print(collection)
        
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

        if isinstance(args, dict) and args.get("file_path"):
            file_path = args["file_path"]
        elif isinstance(args, list) and args and args[0]:
            file_path = args[0]
        else:
            print("No valid file path provided")
            return True

        try:
            document_vector.upload_file(file_path)
        except FileNotFoundError as e:
            print(e)
            return True
            
        print(f"File {file_path} uploaded successfully")
        return True
    
    def _delete_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle delete command"""
        if isinstance(args, dict) and args.get("filename"):
            filename = args["filename"]
        elif isinstance(args, list) and args and args[0]:
            filename = args[0]
        else:
            print("No valid filename provided")
            return True

        try:
            document_vector.delete_rag_file(filename)
        except FileNotFoundError as e:
            print(e)
            return True
        
        print(f"File {filename} and related collections was deleted successfully")
        return True

    def _collections_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        """Handle get collections command"""
        collections = document_vector.show_all_collections()
        if not collections:
            print("Program has not collections")
            return True
        
        print("You have next collections created this program:")
        for collection_name in document_vector.show_all_collections():
            print(collection_name)
        
        return True

    def _chunks_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        "Handle get chunks command"
        
        if isinstance(args, dict) and args.get("filename") and args.get("query"):
            filename = args["filename"]
            query = args["query"]
        elif isinstance(args, list) and args and args[0] and args[1]:
            filename = args[0]
            query = args[1]
        else:
            print("No valid filename or query provided")
            return True
        
        collections = document_vector.find_chunks_from_file(filename, query)
        
        for num, collection in enumerate(collections, 1):
            print(f"{num} relevant chunk:\n")
            print(str(collection) + '\n\n')

        return True

    def _result_command(self, args: Optional[Union[Dict[str, str], List[str]]] = None):
        "Handle get rag results command"
        if isinstance(args, dict) and args.get("filename") and args.get("query"):
            filename = args["filename"]
            query = args["query"]
        elif isinstance(args, list) and args and args[0] and args[1]:
            filename = args[0]
            query = args[1]
        else:
            print("No valid filename and query provided")
            return True
        
        response = document_vector.rag_search_from_file(filename, query)
        
        print()
        print("Response:")
        print(response)

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
            command_name = parts[0]
            
            if len(parts) == 1:
                args = None
            else:
                string_args = " ".join(parts[1:])
                args = handler.parser_args(string_args)

            is_active = handler.execute_command(command_name, args)

        except CliArgumentsError as e:
            print(f"\n{e}")
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