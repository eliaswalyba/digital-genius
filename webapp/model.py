from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer
from sklearn.pipeline import Pipeline

class Model():
    
    def __init__(self, 
        classifier, 
        logreg_jobs=1, 
        logreg_c=1e5, 
        lsvm_loss='hinge', 
        lsvm_penalty='l2', 
        lsvm_alpha=1e-3, 
        lsvm_random_state=42, 
        lsvm_max_iter=5, 
        lsvm_tol=None
    ):
        if classifier == 'naive bayes':
            self.classifier = MultinomialNB()
        elif classifier == 'logistic regression':
            self.classifier = LogisticRegression(n_jobs=logreg_jobs, C=logreg_c)
        elif classifier == 'lsvm':
            self.classifier = SGDClassifier(
                loss=lsvm_loss, 
                penalty=lsvm_penalty, 
                alpha=lsvm_alpha, 
                random_state=lsvm_random_state, 
                max_iter=lsvm_max_iter, 
                tol=lsvm_tol
            )
        else:
            self.classifier = None
            
            
    def train(self, feature, label):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('transformer', TfidfTransformer()),
            ('classifier', self.classifier),
        ]).fit(feature, label)
        
        return self
    
    def test(self, test):
        return self.model.predict(test)