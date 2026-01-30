"""Business impact simulator for supply chain disruptions."""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import log
from src.utils.config import config


class BusinessImpactSimulator:
    """Calculate business impact of supply chain disruptions."""
    
    def __init__(self):
        """Initialize business impact simulator."""
        self.business_config = config.business
        
    def estimate_revenue_loss(self, predictions_df: pd.DataFrame,
                             order_values: pd.Series) -> pd.DataFrame:
        """
        Estimate revenue loss from delays.
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            order_values: Series with order values
            
        Returns:
            DataFrame with revenue loss estimates
        """
        # Calculate expected loss based on delay probability
        expected_loss = order_values * predictions_df['delay_probability']
        
        # Calculate worst-case loss (if all predicted delays occur)
        worst_case_loss = order_values * predictions_df['prediction']
        
        # Create results DataFrame
        loss_df = pd.DataFrame({
            'order_value': order_values,
            'delay_probability': predictions_df['delay_probability'],
            'expected_loss': expected_loss,
            'worst_case_loss': worst_case_loss,
            'risk_category': predictions_df['risk_category']
        })
        
        log.info(f"Total expected revenue loss: ${expected_loss.sum():,.2f}")
        log.info(f"Total worst-case revenue loss: ${worst_case_loss.sum():,.2f}")
        
        return loss_df
    
    def calculate_sla_breach_risk(self, predictions_df: pd.DataFrame,
                                  order_values: pd.Series) -> pd.DataFrame:
        """
        Calculate SLA breach risk and penalties.
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            order_values: Series with order values
            
        Returns:
            DataFrame with SLA breach risk
        """
        penalty_rate = self.business_config.sla_penalty_rate
        
        # Calculate SLA penalty (percentage of order value)
        sla_penalty = order_values * penalty_rate * predictions_df['delay_probability']
        
        # Calculate breach probability
        breach_probability = predictions_df['delay_probability']
        
        sla_df = pd.DataFrame({
            'order_value': order_values,
            'breach_probability': breach_probability,
            'expected_penalty': sla_penalty,
            'max_penalty': order_values * penalty_rate * predictions_df['prediction'],
            'risk_category': predictions_df['risk_category']
        })
        
        log.info(f"Total expected SLA penalties: ${sla_penalty.sum():,.2f}")
        
        return sla_df
    
    def estimate_churn_risk(self, predictions_df: pd.DataFrame,
                           customer_ids: pd.Series = None) -> pd.DataFrame:
        """
        Estimate customer churn risk from delays.
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            customer_ids: Series with customer IDs (optional)
            
        Returns:
            DataFrame with churn risk
        """
        churn_multiplier = self.business_config.churn_probability_multiplier
        customer_ltv = self.business_config.customer_lifetime_value
        
        # Calculate churn probability (increases with delay probability)
        churn_probability = predictions_df['delay_probability'] * churn_multiplier
        churn_probability = np.clip(churn_probability, 0, 1)
        
        # Calculate expected LTV loss
        expected_ltv_loss = churn_probability * customer_ltv
        
        churn_df = pd.DataFrame({
            'delay_probability': predictions_df['delay_probability'],
            'churn_probability': churn_probability,
            'expected_ltv_loss': expected_ltv_loss,
            'risk_category': predictions_df['risk_category']
        })
        
        if customer_ids is not None:
            churn_df['customer_id'] = customer_ids
        
        log.info(f"Total expected LTV loss from churn: ${expected_ltv_loss.sum():,.2f}")
        log.info(f"Customers at high churn risk: {(churn_probability > 0.1).sum()}")
        
        return churn_df
    
    def calculate_total_impact(self, predictions_df: pd.DataFrame,
                              order_values: pd.Series,
                              customer_ids: pd.Series = None) -> Dict:
        """
        Calculate total business impact across all metrics.
        
        Args:
            predictions_df: DataFrame with predictions and probabilities
            order_values: Series with order values
            customer_ids: Series with customer IDs (optional)
            
        Returns:
            Dictionary with comprehensive impact metrics
        """
        # Calculate individual impacts
        revenue_loss = self.estimate_revenue_loss(predictions_df, order_values)
        sla_breach = self.calculate_sla_breach_risk(predictions_df, order_values)
        churn_risk = self.estimate_churn_risk(predictions_df, customer_ids)
        
        # Aggregate metrics
        total_impact = {
            'revenue': {
                'expected_loss': revenue_loss['expected_loss'].sum(),
                'worst_case_loss': revenue_loss['worst_case_loss'].sum(),
                'at_risk_orders': (predictions_df['prediction'] == 1).sum()
            },
            'sla': {
                'expected_penalty': sla_breach['expected_penalty'].sum(),
                'max_penalty': sla_breach['max_penalty'].sum(),
                'high_risk_orders': (sla_breach['breach_probability'] > 0.7).sum()
            },
            'churn': {
                'expected_ltv_loss': churn_risk['expected_ltv_loss'].sum(),
                'high_risk_customers': (churn_risk['churn_probability'] > 0.1).sum(),
                'avg_churn_probability': churn_risk['churn_probability'].mean()
            },
            'overall': {
                'total_expected_loss': (
                    revenue_loss['expected_loss'].sum() +
                    sla_breach['expected_penalty'].sum() +
                    churn_risk['expected_ltv_loss'].sum()
                ),
                'orders_analyzed': len(predictions_df),
                'high_risk_orders': (predictions_df['risk_category'] == 'High').sum(),
                'critical_risk_orders': (predictions_df['risk_category'] == 'Critical').sum()
            }
        }
        
        log.info(f"Total expected business impact: ${total_impact['overall']['total_expected_loss']:,.2f}")
        
        return total_impact
    
    def generate_impact_summary(self, total_impact: Dict) -> pd.DataFrame:
        """
        Generate summary table of business impact.
        
        Args:
            total_impact: Dictionary with impact metrics
            
        Returns:
            DataFrame with formatted summary
        """
        summary_data = [
            {
                'Metric': 'Expected Revenue Loss',
                'Value': f"${total_impact['revenue']['expected_loss']:,.2f}",
                'Category': 'Revenue'
            },
            {
                'Metric': 'Worst-Case Revenue Loss',
                'Value': f"${total_impact['revenue']['worst_case_loss']:,.2f}",
                'Category': 'Revenue'
            },
            {
                'Metric': 'Orders at Risk',
                'Value': str(total_impact['revenue']['at_risk_orders']),
                'Category': 'Revenue'
            },
            {
                'Metric': 'Expected SLA Penalties',
                'Value': f"${total_impact['sla']['expected_penalty']:,.2f}",
                'Category': 'SLA'
            },
            {
                'Metric': 'High-Risk SLA Breaches',
                'Value': str(total_impact['sla']['high_risk_orders']),
                'Category': 'SLA'
            },
            {
                'Metric': 'Expected LTV Loss (Churn)',
                'Value': f"${total_impact['churn']['expected_ltv_loss']:,.2f}",
                'Category': 'Churn'
            },
            {
                'Metric': 'High-Risk Customers',
                'Value': str(total_impact['churn']['high_risk_customers']),
                'Category': 'Churn'
            },
            {
                'Metric': 'Total Expected Loss',
                'Value': f"${total_impact['overall']['total_expected_loss']:,.2f}",
                'Category': 'Overall'
            },
            {
                'Metric': 'Critical Risk Orders',
                'Value': str(total_impact['overall']['critical_risk_orders']),
                'Category': 'Overall'
            }
        ]
        
        return pd.DataFrame(summary_data)
    
    def create_mitigation_recommendations(self, predictions_df: pd.DataFrame,
                                         top_n: int = 10) -> List[Dict]:
        """
        Generate recommendations for high-risk orders.
        
        Args:
            predictions_df: DataFrame with predictions
            top_n: Number of top recommendations
            
        Returns:
            List of recommendations
        """
        # Sort by risk
        high_risk = predictions_df.nlargest(top_n, 'delay_probability')
        
        recommendations = []
        for idx, row in high_risk.iterrows():
            rec = {
                'order_index': idx,
                'delay_probability': row['delay_probability'],
                'risk_category': row['risk_category'],
                'recommendation': self._get_recommendation(row['risk_category'])
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _get_recommendation(self, risk_category: str) -> str:
        """Get recommendation based on risk category."""
        recommendations = {
            'Low': 'Monitor regularly',
            'Medium': 'Increase monitoring frequency, prepare contingency',
            'High': 'Immediate review required, activate backup suppliers',
            'Critical': 'URGENT: Escalate to management, implement emergency measures'
        }
        return recommendations.get(risk_category, 'Review needed')
