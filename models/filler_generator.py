import random
import re

class FillerGenerator:
    
    def __init__(self):
        self.filler_templates = {
            'acknowledgment': [
                "I see, that's interesting...",
                "I understand...",
                "That makes sense...",
                "Interesting point..."
            ],
            'thinking': [
                "Let me think about that...",
                "Hmm, let me consider...",
                "That's worth exploring..."
            ],
            'clarification': [
                "Could you elaborate on that?",
                "Can you tell me more about that?",
                "What do you mean by that exactly?",
                "Could you explain that further?"
            ],
            'technical': [
                "That's a solid technical approach...",
                "How did you implement that?",
                "What technologies did you use?",
                "Tell me more about the architecture..."
            ],
            'challenge': [
                "How did you overcome that challenge?",
                "What was difficult about that?",
                "What alternatives did you consider?",
                "How did you solve that problem?"
            ]
        }
        
        self.context_keywords = {
            'technical': ['code', 'function', 'algorithm', 'implementation', 'built', 'developed'],
            'challenge': ['problem', 'issue', 'difficult', 'challenge', 'struggle', 'error'],
            'explanation': ['because', 'reason', 'therefore', 'so', 'why'],
            'project': ['project', 'application', 'system', 'tool', 'platform']
        }
    
    def detect_context(self, text):
        text_lower = text.lower()
        
        for context, keywords in self.context_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return context
        
        return 'acknowledgment'
    
    def generate(self, text, context=None):
        if context is None:
            context = self.detect_context(text)
        
        # Map detected context to filler category
        category_map = {
            'technical': 'technical',
            'challenge': 'challenge',
            'explanation': 'thinking',
            'project': 'clarification'
        }
        
        category = category_map.get(context, 'acknowledgment')
        
        # Randomly select from category
        fillers = self.filler_templates.get(category, self.filler_templates['acknowledgment'])
        return random.choice(fillers)
    
    def detect_pause(self, text, threshold=2.0):
        # Count sentences
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) >= 2 or len(text) > 200:
            return True
        
        return False