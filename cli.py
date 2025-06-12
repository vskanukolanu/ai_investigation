import asyncio
import click
from typing import List, Dict
from workflow.investigation_manager import InvestigationManager, MetricAnomaly
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
import json

console = Console()

def print_hypotheses(hypotheses: List[Dict]) -> None:
    """Display ranked hypotheses in a table"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID")
    table.add_column("Description")
    table.add_column("Score")
    table.add_column("Confidence")
    
    for h in hypotheses:
        table.add_row(
            h['id'],
            h['description'],
            f"{h['scores'].final_score:.2f}",
            f"{h.get('success_rate', 0):.2f}"
        )
    
    console.print(table)

def get_hypothesis_details() -> Dict:
    """Gather details for a new hypothesis"""
    console.print("\n[bold]Creating New Hypothesis[/bold]")
    
    details = {
        'name': Prompt.ask("Hypothesis name"),
        'description': Prompt.ask("Description"),
        'query_template': Prompt.ask("Query template name"),
        'metrics': Prompt.ask("Required metrics (comma-separated)").split(','),
        'params': {}
    }
    
    # Get optional parameters
    while Confirm.ask("Add parameter?"):
        param_name = Prompt.ask("Parameter name")
        param_value = Prompt.ask("Parameter value")
        details['params'][param_name] = param_value
    
    return details

@click.group()
def cli():
    """Investigation Agent CLI"""
    pass

@cli.command()
@click.option('--metric', required=True, help='Metric showing anomaly')
@click.option('--change', required=True, type=float, help='Percentage change')
@click.option('--period', default='week', help='Time period')
@click.option('--groups', help='Affected groups (comma-separated)')
@click.option('--keywords', help='Context keywords (comma-separated)')
def investigate(metric: str, change: float, period: str, groups: str, keywords: str):
    """Start an investigation for a metric anomaly"""
    # Initialize manager
    manager = InvestigationManager(graph=None)  # You'd initialize with your graph
    
    # Create anomaly object
    anomaly = MetricAnomaly(
        metric_name=metric,
        change_percentage=change,
        time_period=period,
        groups=groups.split(',') if groups else None,
        keywords=keywords.split(',') if keywords else None
    )
    
    # Get initial hypotheses
    investigation = manager.initiate_investigation(anomaly)
    
    console.print(f"\n[bold]Investigation Started: {investigation['investigation_id']}[/bold]")
    console.print("\nRelevant Hypotheses:")
    print_hypotheses(investigation['hypotheses'])
    
    # Ask if user wants to add new hypothesis
    if Confirm.ask("\nWould you like to add a new hypothesis?"):
        details = get_hypothesis_details()
        new_id = manager.add_new_hypothesis(details)
        console.print(f"\nCreated new hypothesis: {new_id}")
        
        # Refresh hypotheses list
        investigation = manager.initiate_investigation(anomaly)
        console.print("\nUpdated Hypotheses:")
        print_hypotheses(investigation['hypotheses'])
    
    # Select hypotheses to test
    selected_ids = Prompt.ask(
        "\nEnter hypothesis IDs to test (comma-separated)",
        default=','.join(h['id'] for h in investigation['hypotheses'][:3])
    ).split(',')
    
    # Execute selected hypotheses
    console.print("\n[bold]Executing Hypotheses...[/bold]")
    results = asyncio.run(
        manager.execute_hypotheses(investigation['investigation_id'], selected_ids)
    )
    
    # Display results
    console.print("\n[bold]Investigation Results:[/bold]")
    console.print(json.dumps(results, indent=2))
    
    # Gather feedback
    console.print("\n[bold]Provide Feedback:[/bold]")
    for hypothesis_id in selected_ids:
        rating = int(Prompt.ask(
            f"\nRate hypothesis {hypothesis_id} (1-5)",
            choices=['1', '2', '3', '4', '5']
        ))
        comments = Prompt.ask("Additional comments (optional)")
        
        manager.record_user_feedback(
            investigation['investigation_id'],
            hypothesis_id,
            rating,
            comments
        )
    
    console.print("\n[bold green]Investigation Complete![/bold green]")

if __name__ == '__main__':
    cli() 