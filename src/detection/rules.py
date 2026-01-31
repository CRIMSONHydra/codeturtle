"""
Rule-Based Risk Detection

Static analysis rules to detect common code smells,
anti-patterns, and potential bugs.
"""

import ast
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RISK_THRESHOLDS, RISK_WEIGHTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskSeverity(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFinding:
    """A single risk finding in code."""
    rule_id: str
    rule_name: str
    severity: RiskSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    snippet: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'line_number': self.line_number,
            'column': self.column,
        }


@dataclass
class RiskReport:
    """Complete risk analysis report for a code file."""
    filepath: Optional[str]
    findings: List[RiskFinding]
    risk_score: float  # 0-100
    summary: Dict[str, int]  # Count by severity
    
    def has_critical(self) -> bool:
        return any(f.severity == RiskSeverity.CRITICAL for f in self.findings)
    
    def has_high(self) -> bool:
        return any(f.severity == RiskSeverity.HIGH for f in self.findings)


class RiskDetector:
    """
    Rule-based code risk detector.
    
    Analyzes Python code for common anti-patterns and risks.
    """
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize detector with risk thresholds.
        
        Args:
            thresholds: Custom thresholds (default: from settings)
        """
        self.thresholds = thresholds or RISK_THRESHOLDS
    
    def analyze(self, code: str, filepath: Optional[str] = None) -> RiskReport:
        """
        Analyze code for risks.
        
        Args:
            code: Python source code
            filepath: Optional filepath for reporting
            
        Returns:
            RiskReport with all findings
        """
        findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return RiskReport(
                filepath=filepath,
                findings=[RiskFinding(
                    rule_id='SYNTAX_ERROR',
                    rule_name='Syntax Error',
                    severity=RiskSeverity.CRITICAL,
                    message=f"Code has syntax error: {e}",
                    line_number=e.lineno,
                )],
                risk_score=100.0,
                summary={'critical': 1},
            )
        
        # Run all rule checks
        findings.extend(self._check_bare_except(tree))
        findings.extend(self._check_deep_nesting(tree, code))
        findings.extend(self._check_long_functions(tree))
        findings.extend(self._check_too_many_arguments(tree))
        findings.extend(self._check_global_usage(tree))
        findings.extend(self._check_mutable_defaults(tree))
        findings.extend(self._check_recursion_without_base(tree))
        findings.extend(self._check_broad_exception_types(tree))
        findings.extend(self._check_unused_variables(tree))
        findings.extend(self._check_magic_numbers(tree))
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(findings)
        
        # Summary by severity
        summary = {s.value: 0 for s in RiskSeverity}
        for f in findings:
            summary[f.severity.value] += 1
        
        return RiskReport(
            filepath=filepath,
            findings=findings,
            risk_score=risk_score,
            summary=summary,
        )
    
    def _check_bare_except(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for bare except clauses."""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    findings.append(RiskFinding(
                        rule_id='BARE_EXCEPT',
                        rule_name='Bare Except Clause',
                        severity=RiskSeverity.HIGH,
                        message="Bare 'except:' catches all exceptions including KeyboardInterrupt and SystemExit. Use 'except Exception:' instead.",
                        line_number=node.lineno,
                    ))
        
        return findings
    
    def _check_deep_nesting(self, tree: ast.AST, code: str) -> List[RiskFinding]:
        """Check for deeply nested code."""
        findings = []
        max_depth = self.thresholds.get('max_nesting_depth', 5)
        
        class NestingChecker(ast.NodeVisitor):
            def __init__(self):
                self.current_depth = 0
                self.deep_nodes = []
            
            def _visit_nesting(self, node):
                self.current_depth += 1
                if self.current_depth > max_depth:
                    self.deep_nodes.append((node, self.current_depth))
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_For(self, node): self._visit_nesting(node)
            def visit_While(self, node): self._visit_nesting(node)
            def visit_If(self, node): self._visit_nesting(node)
            def visit_With(self, node): self._visit_nesting(node)
            def visit_Try(self, node): self._visit_nesting(node)
        
        checker = NestingChecker()
        checker.visit(tree)
        
        for node, depth in checker.deep_nodes:
            findings.append(RiskFinding(
                rule_id='DEEP_NESTING',
                rule_name='Deep Nesting',
                severity=RiskSeverity.MEDIUM,
                message=f"Code is nested {depth} levels deep. Consider refactoring to reduce complexity.",
                line_number=node.lineno,
            ))
        
        return findings
    
    def _check_long_functions(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for overly long functions."""
        findings = []
        max_length = self.thresholds.get('max_function_length', 50)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno:
                    length = node.end_lineno - node.lineno + 1
                    if length > max_length:
                        findings.append(RiskFinding(
                            rule_id='LONG_FUNCTION',
                            rule_name='Long Function',
                            severity=RiskSeverity.LOW,
                            message=f"Function '{node.name}' is {length} lines long. Consider breaking it into smaller functions.",
                            line_number=node.lineno,
                        ))
        
        return findings
    
    def _check_too_many_arguments(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for functions with too many arguments."""
        findings = []
        max_args = self.thresholds.get('max_parameters', 7)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_args = (
                    len(node.args.args) + 
                    len(node.args.kwonlyargs) +
                    (1 if node.args.vararg else 0) +
                    (1 if node.args.kwarg else 0)
                )
                # Subtract 'self' or 'cls' for methods
                if node.args.args and node.args.args[0].arg in ('self', 'cls'):
                    total_args -= 1
                
                if total_args > max_args:
                    findings.append(RiskFinding(
                        rule_id='TOO_MANY_ARGS',
                        rule_name='Too Many Arguments',
                        severity=RiskSeverity.LOW,
                        message=f"Function '{node.name}' has {total_args} parameters. Consider using a config object or dataclass.",
                        line_number=node.lineno,
                    ))
        
        return findings
    
    def _check_global_usage(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for global variable usage."""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                findings.append(RiskFinding(
                    rule_id='GLOBAL_USAGE',
                    rule_name='Global Variable Usage',
                    severity=RiskSeverity.MEDIUM,
                    message=f"Using global variables ({', '.join(node.names)}) makes code harder to test and reason about.",
                    line_number=node.lineno,
                ))
        
        return findings
    
    def _check_mutable_defaults(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for mutable default arguments."""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(RiskFinding(
                            rule_id='MUTABLE_DEFAULT',
                            rule_name='Mutable Default Argument',
                            severity=RiskSeverity.HIGH,
                            message=f"Function '{node.name}' has a mutable default argument. This can cause unexpected behavior.",
                            line_number=node.lineno,
                        ))
                        break
        
        return findings
    
    def _check_recursion_without_base(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for potentially infinite recursion."""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if function calls itself
                calls_self = False
                has_conditional_return = False
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == node.name:
                            calls_self = True
                    if isinstance(child, ast.If):
                        for subchild in ast.walk(child):
                            if isinstance(subchild, ast.Return):
                                has_conditional_return = True
                                break
                
                if calls_self and not has_conditional_return:
                    findings.append(RiskFinding(
                        rule_id='RECURSION_NO_BASE',
                        rule_name='Recursion Without Base Case',
                        severity=RiskSeverity.HIGH,
                        message=f"Recursive function '{node.name}' may not have a proper base case.",
                        line_number=node.lineno,
                    ))
        
        return findings
    
    def _check_broad_exception_types(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for overly broad exception handling."""
        findings = []
        broad_exceptions = {'Exception', 'BaseException'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type and isinstance(node.type, ast.Name):
                    if node.type.id in broad_exceptions:
                        # Check if they're just logging/re-raising
                        just_reraises = False
                        for child in ast.walk(node):
                            if isinstance(child, ast.Raise) and child.exc is None:
                                just_reraises = True
                                break
                        
                        if not just_reraises:
                            findings.append(RiskFinding(
                                rule_id='BROAD_EXCEPTION',
                                rule_name='Broad Exception Handler',
                                severity=RiskSeverity.LOW,
                                message=f"Catching '{node.type.id}' is too broad. Consider catching specific exception types.",
                                line_number=node.lineno,
                            ))
        
        return findings
    
    def _check_unused_variables(self, tree: ast.AST) -> List[RiskFinding]:
        """Simple check for obviously unused variables."""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned = set()
                used = set()
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                if not target.id.startswith('_'):
                                    assigned.add(target.id)
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        used.add(child.id)
                
                unused = assigned - used
                for var in unused:
                    # Skip common false positives
                    if var not in ('self', 'cls', 'args', 'kwargs'):
                        findings.append(RiskFinding(
                            rule_id='UNUSED_VARIABLE',
                            rule_name='Unused Variable',
                            severity=RiskSeverity.LOW,
                            message=f"Variable '{var}' is assigned but never used in function '{node.name}'.",
                            line_number=node.lineno,
                        ))
        
        return findings
    
    def _check_magic_numbers(self, tree: ast.AST) -> List[RiskFinding]:
        """Check for magic numbers (unexplained numeric literals)."""
        findings = []
        allowed = {0, 1, 2, -1, 100, 10}  # Common acceptable values
        
        # Only check inside functions
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(func):
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        if abs(node.value) not in allowed and abs(node.value) > 2:
                            findings.append(RiskFinding(
                                rule_id='MAGIC_NUMBER',
                                rule_name='Magic Number',
                                severity=RiskSeverity.LOW,
                                message=f"Magic number {node.value} should be extracted to a named constant.",
                                line_number=node.lineno if hasattr(node, 'lineno') else None,
                            ))
        
        return findings[:5]  # Limit magic number findings
    
    def _calculate_risk_score(self, findings: List[RiskFinding]) -> float:
        """
        Calculate overall risk score from findings.
        
        Uses logarithmic scaling to avoid ceiling effect at 100:
        - Few findings: linear growth
        - Many findings: diminishing returns (logarithmic)
        """
        if not findings:
            return 0.0
        
        severity_weights = {
            RiskSeverity.LOW: 5,
            RiskSeverity.MEDIUM: 15,
            RiskSeverity.HIGH: 30,
            RiskSeverity.CRITICAL: 50,
        }
        
        raw_score = sum(severity_weights[f.severity] for f in findings)
        
        # Apply logarithmic scaling to avoid ceiling effect
        # Formula: score = 100 * (1 - e^(-raw_score / 50))
        # This gives ~63% at raw=50, ~86% at raw=100, ~95% at raw=150
        import math
        scaled_score = 100 * (1 - math.exp(-raw_score / 50))
        
        return round(scaled_score, 1)


def check_code_risks(code: str, filepath: Optional[str] = None) -> RiskReport:
    """
    Convenience function to check code for risks.
    
    Args:
        code: Python source code
        filepath: Optional filepath for reporting
        
    Returns:
        RiskReport with all findings
    """
    detector = RiskDetector()
    return detector.analyze(code, filepath)


if __name__ == "__main__":
    # Test risk detection
    risky_code = '''
def process(data, items=[], config={}):
    """Problematic function."""
    global counter
    counter += 1
    
    try:
        for item in items:
            for sub in item:
                for x in sub:
                    for y in x:
                        for z in y:
                            if z > 0:
                                result = z * 3.14159
    except:
        pass
    
    return process(data[1:])  # No base case!

def compute(a, b, c, d, e, f, g, h, i, j):
    """Too many arguments."""
    return a + b + 42
'''
    
    report = check_code_risks(risky_code)
    
    print(f"\nRisk Score: {report.risk_score:.1f}/100")
    print(f"Findings: {len(report.findings)}")
    print(f"Summary: {report.summary}")
    print("\nDetailed Findings:")
    for f in report.findings:
        print(f"  [{f.severity.value.upper()}] Line {f.line_number}: {f.message[:60]}...")
