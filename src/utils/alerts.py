"""Alert system for supply chain disruptions."""
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF

from src.utils.logger import log
from src.utils.config import config


class AlertSystem:
    """Generate and manage alerts for high-risk predictions."""
    
    def __init__(self):
        """Initialize alert system."""
        self.alert_config = config.alerts
        self.alerts = []
        
    def check_thresholds(self, predictions_df: pd.DataFrame) -> List[Dict]:
        """
        Check predictions against alert thresholds.
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            
        Returns:
            List of alerts triggered
        """
        threshold = self.alert_config.get('high_risk_threshold', 0.7)
        
        # Find high-risk predictions
        high_risk = predictions_df[predictions_df['delay_probability'] >= threshold]
        
        alerts = []
        for idx, row in high_risk.iterrows():
            alert = {
                'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
                'timestamp': datetime.now().isoformat(),
                'order_index': idx,
                'delay_probability': row['delay_probability'],
                'risk_category': row['risk_category'],
                'severity': self._get_severity(row['delay_probability']),
                'message': self._generate_alert_message(row)
            }
            alerts.append(alert)
        
        self.alerts.extend(alerts)
        log.info(f"Generated {len(alerts)} alerts (threshold: {threshold})")
        
        return alerts
    
    def _get_severity(self, probability: float) -> str:
        """Determine alert severity based on probability."""
        if probability >= 0.9:
            return 'CRITICAL'
        elif probability >= 0.8:
            return 'HIGH'
        elif probability >= 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_alert_message(self, row: pd.Series) -> str:
        """Generate alert message."""
        return (
            f"High risk of delay detected: {row['delay_probability']:.1%} probability. "
            f"Risk category: {row['risk_category']}. Immediate attention required."
        )
    
    def generate_alert_report(self, predictions_df: pd.DataFrame,
                            business_impact: Dict,
                            top_risks: pd.DataFrame) -> str:
        """
        Generate comprehensive alert report.
        
        Args:
            predictions_df: DataFrame with predictions
            business_impact: Business impact metrics
            top_risks: DataFrame with top risk drivers
            
        Returns:
            Report text
        """
        report = []
        report.append("="*80)
        report.append("SUPPLY CHAIN DISRUPTION ALERT REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal Orders Analyzed: {len(predictions_df)}")
        
        # Risk summary
        report.append("\n" + "-"*80)
        report.append("RISK SUMMARY")
        report.append("-"*80)
        risk_counts = predictions_df['risk_category'].value_counts()
        for category, count in risk_counts.items():
            report.append(f"{category} Risk: {count} orders ({count/len(predictions_df):.1%})")
        
        # Business impact
        report.append("\n" + "-"*80)
        report.append("BUSINESS IMPACT")
        report.append("-"*80)
        report.append(f"Expected Revenue Loss: ${business_impact['revenue']['expected_loss']:,.2f}")
        report.append(f"Expected SLA Penalties: ${business_impact['sla']['expected_penalty']:,.2f}")
        report.append(f"Expected LTV Loss: ${business_impact['churn']['expected_ltv_loss']:,.2f}")
        report.append(f"Total Expected Impact: ${business_impact['overall']['total_expected_loss']:,.2f}")
        
        # Top risks
        report.append("\n" + "-"*80)
        report.append("TOP 10 HIGH-RISK ORDERS")
        report.append("-"*80)
        high_risk_orders = predictions_df.nlargest(10, 'delay_probability')
        for idx, row in high_risk_orders.iterrows():
            report.append(
                f"Order #{idx}: {row['delay_probability']:.1%} probability, "
                f"Risk: {row['risk_category']}"
            )
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def export_to_text(self, report: str, filename: Optional[str] = None) -> str:
        """
        Export report to text file.
        
        Args:
            report: Report text
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alert_report_{timestamp}.txt"
        
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)
        
        log.info(f"Report exported to {filepath}")
        return str(filepath)
    
    def export_to_pdf(self, report: str, filename: Optional[str] = None) -> str:
        """
        Export report to PDF file.
        
        Args:
            report: Report text
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alert_report_{timestamp}.pdf"
        
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Supply Chain Disruption Alert Report", 0, 1, 'C')
        
        pdf.set_font("Arial", size=10)
        for line in report.split('\n'):
            if line.strip():
                pdf.multi_cell(0, 5, line)
        
        pdf.output(str(filepath))
        log.info(f"PDF report exported to {filepath}")
        
        return str(filepath)
    
    def export_alerts_to_csv(self, alerts: List[Dict], filename: Optional[str] = None) -> str:
        """
        Export alerts to CSV file.
        
        Args:
            alerts: List of alerts
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alerts_{timestamp}.csv"
        
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        alerts_df = pd.DataFrame(alerts)
        alerts_df.to_csv(filepath, index=False)
        
        log.info(f"Alerts exported to {filepath}")
        return str(filepath)
    
    def send_email_alert(self, alert: Dict) -> bool:
        """
        Send email alert (placeholder for future implementation).
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Success status
        """
        # Placeholder - would integrate with SMTP in production
        log.info(f"Email alert sent: {alert['alert_id']}")
        return True
    
    def send_slack_notification(self, alert: Dict) -> bool:
        """
        Send Slack notification (placeholder for future implementation).
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Success status
        """
        # Placeholder - would integrate with Slack webhook in production
        log.info(f"Slack notification sent: {alert['alert_id']}")
        return True
