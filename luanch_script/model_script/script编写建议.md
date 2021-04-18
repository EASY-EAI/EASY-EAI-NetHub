model_script_field 模型脚本攥写建议：

```python
class model_script(object):
    def __init__(self, model, loss_function, train_loader, eval_loader, test_loader, 					 optiomizer, device='cpu'):
    	super(model_script, self).__init__()
        ...
        
    def train(self, epoch):
        ...
        
    def eval(self):
        ...
        
    def validate_on_test(self):
        ...
        
    def save_model(self, path):
        ...
    
    def load_model(self, path):
        ...
        
    def set_eval(self, path):
        ...
        
    def set_train(self, path):
        ...
        
    def _to_device(self, device):
        ...
        
    def export_frozen_model(self, method):
        ...
        
        
        
        
```

