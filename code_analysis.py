import ast
import re

class CodeAnalysisPipeline:
    def __init__(self):
        self.generic_names = {
            'i', 'j', 'k', 'n', 'm', 'x', 'y', 'z', 'data', 'temp', 'tmp', 
            'result', 'res', 'idx', 'val', 'value', 'item', 'elem', 'element', 
            'arr', 'array', 'lst', 'list', 'num', 'count', 'param', 'args', 
            'kwargs', 'arg', 'obj', 'entry', 'info', 'msg', 'message', 'status'
        }
    
    def extract_code_metrics(self, code_string):
        """Extract basic code structure metrics"""
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return None
        
        visitor = CodeVisitor()
        visitor.visit(tree)
        
        lines = code_string.splitlines()
        total_lines = len(lines)
        
        # Count comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Extract docstrings
        docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node):
                    docstrings.append(ast.get_docstring(node))
        
        return {
            'total_lines': total_lines,
            'num_functions': visitor.functions,
            'num_classes': visitor.classes,
            'max_nesting_depth': visitor.max_depth,
            'imported_modules': list(visitor.imports),
            'defined_variables': list(visitor.variables),
            'comment_lines': comment_lines,
            'num_docstrings': len(docstrings),
            'docstrings': docstrings
        }
    
    def analyze_variable_patterns(self, variables):
        """Analyze variable naming patterns"""
        if not variables:
            return {'generic_ratio': 0, 'avg_length': 0, 'total_vars': 0}
        
        generic_count = sum(1 for var in variables if var in self.generic_names)
        avg_length = sum(len(var) for var in variables) / len(variables)
        
        return {
            'total_vars': len(variables),
            'generic_count': generic_count,
            'generic_ratio': generic_count / len(variables),
            'avg_length': avg_length
        }
    
    def analyze_comment_patterns(self, docstrings, comment_lines, total_lines):
        """Analyze commenting patterns"""
        if not docstrings and comment_lines == 0:
            return {'comment_ratio': 0, 'avg_docstring_length': 0, 'verbose_docs': False}
        
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        if docstrings:
            avg_doc_length = sum(len(doc) for doc in docstrings) / len(docstrings)
            verbose_docs = avg_doc_length > 200  # Long docstrings
        else:
            avg_doc_length = 0
            verbose_docs = False
        
        return {
            'comment_ratio': comment_ratio,
            'avg_docstring_length': avg_doc_length,
            'verbose_docs': verbose_docs,
            'num_docstrings': len(docstrings)
        }
    
    def detect_ai_patterns(self, metrics):
        """Detect patterns typical of AI-generated code"""
        ai_score = 0
        indicators = []
        
        # High comment ratio with verbose docstrings
        comment_analysis = self.analyze_comment_patterns(
            metrics['docstrings'], metrics['comment_lines'], metrics['total_lines']
        )
        
        if comment_analysis['comment_ratio'] > 0.2:
            ai_score += 2
            indicators.append("High comment ratio")
        
        if comment_analysis['verbose_docs']:
            ai_score += 2
            indicators.append("Verbose docstrings")
        
        # Perfect documentation (all functions have docstrings)
        if metrics['num_functions'] > 0 and comment_analysis['num_docstrings'] == metrics['num_functions']:
            ai_score += 1.5
            indicators.append("All functions documented")
        
        # Variable naming patterns
        var_analysis = self.analyze_variable_patterns(metrics['defined_variables'])
        
        if var_analysis['generic_ratio'] > 0.4:
            ai_score += 1
            indicators.append("High generic variable usage")
        
        if var_analysis['avg_length'] > 12:
            ai_score += 0.5
            indicators.append("Very long variable names")
        
        # Import patterns
        if len(metrics['imported_modules']) > 5:
            ai_score += 0.5
            indicators.append("Many imports")
        
        # Complexity patterns
        if metrics['max_nesting_depth'] > 4:
            ai_score += 1
            indicators.append("High nesting depth")
        
        return {
            'ai_likelihood_score': ai_score,
            'indicators': indicators,
            'assessment': self.get_ai_assessment(ai_score)
        }
    
    def get_ai_assessment(self, score):
        """Convert AI score to qualitative assessment"""
        if score >= 5:
            return "Highly Likely AI-generated"
        elif score >= 3:
            return "Likely AI-generated"
        elif score >= 1:
            return "Possibly AI-generated"
        else:
            return "Unlikely AI-generated"
    
    def calculate_fraud_score(self, ai_analysis):
        """Calculate fraud score based on AI likelihood"""
        ai_score = ai_analysis['ai_likelihood_score']
        
        if ai_score >= 5:
            return 8  # High fraud risk for likely AI code
        elif ai_score >= 3:
            return 5  # Medium fraud risk
        elif ai_score >= 1:
            return 2  # Low fraud risk
        else:
            return 0  # No fraud risk
    
    def analyze(self, code_string):
        """Main code analysis pipeline"""
        results = {
            'status': 'completed',
            'fraud_score': 0,
            'analysis_details': {}
        }
        
        # Extract code metrics
        metrics = self.extract_code_metrics(code_string)
        if metrics is None:
            results['status'] = 'error'
            results['message'] = 'Failed to parse code'
            return results
        
        results['analysis_details']['code_metrics'] = metrics
        
        # AI pattern detection
        ai_analysis = self.detect_ai_patterns(metrics)
        results['analysis_details']['ai_detection'] = ai_analysis
        
        # Calculate fraud score
        results['fraud_score'] = self.calculate_fraud_score(ai_analysis)
        
        return results

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = 0
        self.classes = 0
        self.max_depth = 0
        self.current_depth = 0
        self.imports = set()
        self.variables = set()
    
    def visit_FunctionDef(self, node):
        self.functions += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_ClassDef(self, node):
        self.classes += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.variables.add(node.id)
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_For(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_While(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1