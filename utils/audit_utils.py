"""Audit utilities for financial data validation."""

import json
from collections import defaultdict
from typing import List, Dict, Any
from ..models.data_models import AuditIssue
from ..storage.knowledge_graph import SimpleKnowledgeGraph

class AuditEngine:
    """Engine for running various audit checks on financial data."""
    
    def __init__(self, knowledge_graph: SimpleKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
    
    def run_all_checks(self) -> List[AuditIssue]:
        """
        Run all available audit checks.
        
        Returns:
            List of all audit issues found
        """
        issues = []
        
        # Run individual check methods
        issues.extend(self.check_duplicate_amounts())
        issues.extend(self.check_missing_entities())
        issues.extend(self.check_data_consistency())
        issues.extend(self.check_outliers())
        issues.extend(self.check_account_numbers())
        
        return issues
    
    def check_duplicate_amounts(self) -> List[AuditIssue]:
        """
        Check for potentially duplicate monetary amounts.
        
        Returns:
            List of duplicate amount issues
        """
        issues = []
        
        try:
            # Get all money entities
            money_entities = self.knowledge_graph.query_entities('MONEY')
            amounts = defaultdict(list)
            
            for entity in money_entities:
                try:
                    props = entity.get('properties', {})
                    if isinstance(props, str):
                        props = json.loads(props)
                    
                    amount = props.get('amount')
                    if amount is not None:
                        amounts[amount].append(entity)
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Find duplicates
            for amount, entities in amounts.items():
                if len(entities) > 1:
                    entity_ids = [e['id'] for e in entities]
                    issues.append(AuditIssue(
                        type="potential_duplicate",
                        severity="low",
                        description=f"Amount ${amount} appears {len(entities)} times",
                        entities=entity_ids,
                        recommendations=[
                            "Review transactions for potential duplicates",
                            "Verify if these are legitimate separate transactions"
                        ],
                        metadata={
                            "amount": amount,
                            "occurrence_count": len(entities),
                            "documents": list(set(e.get('source_doc') for e in entities))
                        }
                    ))
        
        except Exception as e:
            issues.append(AuditIssue(
                type="audit_error",
                severity="medium",
                description=f"Error checking duplicate amounts: {str(e)}"
            ))
        
        return issues
    
    def check_missing_entities(self) -> List[AuditIssue]:
        """
        Check for documents that may be missing important entities.
        
        Returns:
            List of missing entity issues
        """
        issues = []
        
        try:
            # Get all documents
            all_entities = self.knowledge_graph.query_entities(limit=10000)
            docs_with_entities = defaultdict(list)
            
            for entity in all_entities:
                source_doc = entity.get('source_doc')
                if source_doc:
                    docs_with_entities[source_doc].append(entity)
            
            # Check each document for completeness
            for doc_id, entities in docs_with_entities.items():
                entity_types = set(e['type'] for e in entities)
                
                # Expected entity types in financial documents
                expected_types = {'MONEY', 'PERSON', 'COMPANY'}
                missing_types = expected_types - entity_types
                
                if missing_types:
                    issues.append(AuditIssue(
                        type="missing_entities",
                        severity="medium",
                        description=f"Document may be missing {', '.join(missing_types)} entities",
                        entities=[doc_id],
                        recommendations=[
                            "Review document for unextracted financial entities",
                            "Consider manual entity annotation"
                        ],
                        metadata={
                            "document_id": doc_id,
                            "missing_types": list(missing_types),
                            "found_types": list(entity_types)
                        }
                    ))
        
        except Exception as e:
            issues.append(AuditIssue(
                type="audit_error",
                severity="medium",
                description=f"Error checking missing entities: {str(e)}"
            ))
        
        return issues
    
    def check_data_consistency(self) -> List[AuditIssue]:
        """
        Check for data consistency issues.
        
        Returns:
            List of consistency issues
        """
        issues = []
        
        try:
            # Check for entities with very low confidence
            low_confidence_entities = self.knowledge_graph.query_entities(
                min_confidence=0.0, limit=1000
            )
            
            very_low_confidence = [e for e in low_confidence_entities if e.get('confidence', 1.0) < 0.3]
            
            if very_low_confidence:
                entity_ids = [e['id'] for e in very_low_confidence]
                issues.append(AuditIssue(
                    type="low_confidence_entities",
                    severity="low",
                    description=f"Found {len(very_low_confidence)} entities with very low confidence",
                    entities=entity_ids[:10],  # Limit to first 10 for readability
                    recommendations=[
                        "Review low-confidence entities for accuracy",
                        "Consider removing or manually verifying these entities"
                    ],
                    metadata={
                        "total_count": len(very_low_confidence),
                        "avg_confidence": sum(e.get('confidence', 0) for e in very_low_confidence) / len(very_low_confidence)
                    }
                ))
            
            # Check for entities with missing properties
            entities_missing_props = [e for e in low_confidence_entities 
                                    if not e.get('properties') and e['type'] in ['MONEY', 'PERCENTAGE']]
            
            if entities_missing_props:
                issues.append(AuditIssue(
                    type="missing_properties",
                    severity="medium",
                    description=f"Found {len(entities_missing_props)} financial entities missing expected properties",
                    entities=[e['id'] for e in entities_missing_props[:5]],
                    recommendations=[
                        "Verify property extraction is working correctly",
                        "Re-process documents if necessary"
                    ]
                ))
        
        except Exception as e:
            issues.append(AuditIssue(
                type="audit_error",
                severity="medium",
                description=f"Error checking data consistency: {str(e)}"
            ))
        
        return issues
    
    def check_outliers(self) -> List[AuditIssue]:
        """
        Check for unusual amounts that might indicate errors.
        
        Returns:
            List of outlier issues
        """
        issues = []
        
        try:
            money_entities = self.knowledge_graph.query_entities('MONEY')
            amounts = []
            
            for entity in money_entities:
                try:
                    props = entity.get('properties', {})
                    if isinstance(props, str):
                        props = json.loads(props)
                    
                    amount = props.get('amount')
                    if amount is not None and isinstance(amount, (int, float)):
                        amounts.append((amount, entity))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
            
            if len(amounts) < 3:  # Need at least 3 amounts for outlier detection
                return issues
            
            # Simple outlier detection using IQR method
            amount_values = [a[0] for a in amounts]
            amount_values.sort()
            
            q1 = amount_values[len(amount_values) // 4]
            q3 = amount_values[3 * len(amount_values) // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [(amount, entity) for amount, entity in amounts 
                       if amount < lower_bound or amount > upper_bound]
            
            if outliers:
                # Focus on extreme outliers (very high amounts)
                extreme_outliers = [(amount, entity) for amount, entity in outliers if amount > upper_bound + iqr]
                
                if extreme_outliers:
                    entity_ids = [e['id'] for _, e in extreme_outliers]
                    max_amount = max(amount for amount, _ in extreme_outliers)
                    
                    issues.append(AuditIssue(
                        type="amount_outliers",
                        severity="medium",
                        description=f"Found {len(extreme_outliers)} unusually high amounts (max: ${max_amount:,.2f})",
                        entities=entity_ids,
                        recommendations=[
                            "Verify these large amounts are correct",
                            "Check for data entry errors or parsing issues"
                        ],
                        metadata={
                            "outlier_count": len(extreme_outliers),
                            "max_amount": max_amount,
                            "upper_bound": upper_bound
                        }
                    ))
        
        except Exception as e:
            issues.append(AuditIssue(
                type="audit_error",
                severity="medium",
                description=f"Error checking outliers: {str(e)}"
            ))
        
        return issues
    
    def check_account_numbers(self) -> List[AuditIssue]:
        """
        Check account numbers for validity and potential issues.
        
        Returns:
            List of account number issues
        """
        issues = []
        
        try:
            account_entities = self.knowledge_graph.query_entities('ACCOUNT_NUMBER')
            
            invalid_accounts = []
            duplicate_accounts = defaultdict(list)
            
            for entity in account_entities:
                try:
                    props = entity.get('properties', {})
                    if isinstance(props, str):
                        props = json.loads(props)
                    
                    clean_number = props.get('clean_number', '')
                    
                    # Check for invalid length
                    if clean_number and (len(clean_number) < 8 or len(clean_number) > 20):
                        invalid_accounts.append(entity)
                    
                    # Check for duplicates
                    if clean_number:
                        duplicate_accounts[clean_number].append(entity)
                
                except (json.JSONDecodeError, KeyError):
                    continue
            
            # Report invalid account numbers
            if invalid_accounts:
                entity_ids = [e['id'] for e in invalid_accounts]
                issues.append(AuditIssue(
                    type="invalid_account_numbers",
                    severity="high",
                    description=f"Found {len(invalid_accounts)} account numbers with invalid length",
                    entities=entity_ids,
                    recommendations=[
                        "Verify account number extraction is correct",
                        "Check source documents for accurate account numbers"
                    ]
                ))
            
            # Report duplicate account numbers
            duplicates = {num: entities for num, entities in duplicate_accounts.items() if len(entities) > 1}
            if duplicates:
                for account_num, entities in list(duplicates.items())[:5]:  # Limit to 5 duplicates
                    entity_ids = [e['id'] for e in entities]
                    issues.append(AuditIssue(
                        type="duplicate_account_numbers",
                        severity="high",
                        description=f"Account number ending in {account_num[-4:]} appears {len(entities)} times",
                        entities=entity_ids,
                        recommendations=[
                            "Verify if this represents the same account across documents",
                            "Consider consolidating duplicate account references"
                        ],
                        metadata={
                            "masked_account": f"****{account_num[-4:]}",
                            "occurrence_count": len(entities)
                        }
                    ))
        
        except Exception as e:
            issues.append(AuditIssue(
                type="audit_error",
                severity="medium",
                description=f"Error checking account numbers: {str(e)}"
            ))
        
        return issues
    
    def generate_audit_report(self, issues: List[AuditIssue]) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report.
        
        Args:
            issues: List of audit issues
            
        Returns:
            Dictionary containing the audit report
        """
        report = {
            "summary": {
                "total_issues": len(issues),
                "high_severity": len([i for i in issues if i.severity == "high"]),
                "medium_severity": len([i for i in issues if i.severity == "medium"]),
                "low_severity": len([i for i in issues if i.severity == "low"])
            },
            "issues_by_type": {},
            "recommendations": set(),
            "issues": []
        }
        
        # Group issues by type
        for issue in issues:
            if issue.type not in report["issues_by_type"]:
                report["issues_by_type"][issue.type] = 0
            report["issues_by_type"][issue.type] += 1
            
            # Collect all recommendations
            report["recommendations"].update(issue.recommendations)
            
            # Add issue to report
            report["issues"].append({
                "type": issue.type,
                "severity": issue.severity,
                "description": issue.description,
                "entity_count": len(issue.entities),
                "recommendations": issue.recommendations,
                "metadata": issue.metadata
            })
        
        # Convert set to list for JSON serialization
        report["recommendations"] = list(report["recommendations"])
        
        return report