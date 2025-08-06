import sys
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    """Enum for message types to avoid string typos"""
    SUCCESS = "success"
    ALERT = "alert" 
    FAILURE = "failure"

class Colors:
    """ANSI color codes for terminal styling"""
    
    GREEN = '\033[92m'      # Bright green
    YELLOW = '\033[93m'     # Bright yellow  
    RED = '\033[91m'        # Bright red
    BLUE = '\033[94m'       # Bright blue
    MAGENTA = '\033[95m'    # Bright magenta
    CYAN = '\033[96m'       # Bright cyan
    WHITE = '\033[97m'      # Bright white
    
    # Styles
    BOLD = '\033[1m'        # Bold text
    UNDERLINE = '\033[4m'   # Underlined text
    RESET = '\033[0m'       # Reset all formatting
    
    # Background colors
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_RED = '\033[101m'



def pretty_print(message_type, message, **kwargs):
    """
    Advanced version with additional formatting options.
    
    Args:
        message_type (str): Type of message
        message (str): The message to be printed
        **kwargs: Additional options:
            - timestamp (bool): Show timestamp
            - bold (bool): Make text bold
            - underline (bool): Underline text
            - background (bool): Use background color
            - custom_icon (str): Custom icon to use
    """
    
    type_config = {
        'success': {
            'color': Colors.GREEN,
            'bg_color': Colors.BG_GREEN,
            'icon': '✅',
            'prefix': 'SUCCESS'
        },
        'alert': {
            'color': Colors.YELLOW,
            'bg_color': Colors.BG_YELLOW,
            'icon': '⚠️',
            'prefix': 'ALERT'
        },
        'failure': {
            'color': Colors.RED,
            'bg_color': Colors.BG_RED,
            'icon': '❌',
            'prefix': 'FAILURE'
        }
    }
    
    message_type = message_type.lower().strip()
    
    if message_type not in type_config:
        print(f"{Colors.RED}❌ Invalid message type: '{message_type}'{Colors.RESET}")
        return
    
    config = type_config[message_type]
    
    
    formatting = ""
    if kwargs.get('bold', False):
        formatting += Colors.BOLD
    if kwargs.get('underline', False):
        formatting += Colors.UNDERLINE
    
    
    color = config['bg_color'] if kwargs.get('background', False) else config['color']
    
    
    icon = kwargs.get('custom_icon', config['icon'])
    
    
    timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] " if kwargs.get('timestamp', False) else ""
    
    
    formatted_message = f"{formatting}{timestamp}{color}{icon} {config['prefix']}: {message}{Colors.RESET}"
    print(formatted_message)



def print_success(message):
    """Quick success message"""
    pretty_print('✅ success', message)

def print_alert(message):
    """Quick alert message"""
    pretty_print(' ⚠️ alert', message)

def print_failure(message):
    """Quick failure message"""
    pretty_print('🛑failure', message)


