"""Load selected fixed weights; never invoke a quantizer during evaluation."""
import json
from .bank import GPTQBank
from .model import load_model


def load_selected_model(bank_path, assignment_path, device='cuda:0'):
    bank = GPTQBank(bank_path)
    model, processor = load_model(bank.manifest['model_path'], device, bank.manifest['dtype'])
    assignment = json.loads(open(assignment_path, encoding='utf-8').read())
    bank.assemble(model, assignment)
    return model, processor, processor.tokenizer
