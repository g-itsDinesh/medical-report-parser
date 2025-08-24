import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
import re

class MedicalReportParser:

    def __init__(self, model_name="d4data/biomedical-ner-all"):
        
        print(f"Loading model '{model_name}'... This may take a moment.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        print("Model loaded successfully.")

    def parse_report(self, report_text: str) -> dict:
        
        inputs = self.tokenizer(report_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        grouped_entities = self._group_entities(tokens, predicted_labels)
        
        categorized_entities = defaultdict(list)
        for entity_text, label in grouped_entities:
            category = label.split('-')[-1]
            categorized_entities[category].append(entity_text)
            
        for category in categorized_entities:
            categorized_entities[category] = list(sorted(set(categorized_entities[category])))

        return dict(categorized_entities)

    def _group_entities(self, tokens: list, labels: list) -> list:
        
        entities = []
        current_entity_tokens = []
        current_entity_label = None

        for token, label in zip(tokens, labels):
            if token in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token):
                continue

            if token.startswith('##'):
                if current_entity_tokens:
                    current_entity_tokens.append(token)
                continue

            if label == 'O':
                if current_entity_tokens:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
                    entities.append((entity_text, current_entity_label))
                    current_entity_tokens = []
                    current_entity_label = None
                continue

            label_category = label.split('-')[-1]
            
            if current_entity_tokens and current_entity_label.split('-')[-1] == label_category:
                current_entity_tokens.append(token)
            else:
                if current_entity_tokens:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
                    entities.append((entity_text, current_entity_label))
                
                current_entity_tokens = [token]
                current_entity_label = 'B-' + label_category

        if current_entity_tokens:
            entity_text = self.tokenizer.convert_tokens_to_string(current_entity_tokens)
            entities.append((entity_text, current_entity_label))
            
        return entities

if __name__ == "__main__":
    parser = MedicalReportParser()
    report_filepath = "report.txt" 

    try:
        with open(report_filepath, 'r', encoding='utf-8') as f:
            report_text = f.read()
        
        print(f"\nSuccessfully read report from '{report_filepath}'.")
        
        print("Parsing report...")
        extracted_data = parser.parse_report(report_text)

        cleaned_data = defaultdict(list)
        junk_tokens = ['-', 'year', 'old', 'grade']
        for category, entities in extracted_data.items():
            for entity in entities:
                
                if entity.strip().lower() in junk_tokens:
                    continue
                
                if category in ['Date', 'Age'] and entity.strip().isdigit() and len(entity.strip()) < 4:
                    continue
                cleaned_data[category].append(entity)


        print("\n--- Extracted Medical Entities ---")
        if not cleaned_data:
            print("No entities were found.")
        else:
            for category, entities in cleaned_data.items():
                print(f"\n[{category}]")
                for entity in entities:
                    print(f"  - {entity}")
        print("\n----------------------------------")

    except FileNotFoundError:
        print(f"Error: The file '{report_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")