from chatbot.gemini_bot import GeminiBot
from rich.console import Console

console = Console()
bot = GeminiBot(name="Gemini Assistant")

console.print("[bold green]Gemini Chatbot đã sẵn sàng![/bold green]")
console.print("Gõ [yellow]'exit'[/yellow] để thoát, [yellow]'clear'[/yellow] để xóa bộ nhớ.\n")

while True:
    user_input = input("Bạn: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    elif user_input.lower() == "clear":
        console.print(f"[red]{bot.clear_context()}[/red]")
        continue

    reply = bot.ask(user_input)
    console.print(f"[bold cyan]{bot.name}:[/bold cyan] {reply}")
