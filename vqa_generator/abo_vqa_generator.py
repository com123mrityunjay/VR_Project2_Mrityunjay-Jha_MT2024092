import os
import json
import random
import base64
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
import google.generativeai as genai
from PIL import Image
import io
import gzip

# Define the question types and difficulty levels for diversity
QUESTION_TYPES = [
    "color", "shape", "material", "size", "pattern", "count", 
    "texture", "position", "function", "brand", "comparison"
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

class ABODataset:
    def __init__(self, abo_base_dir: str, subset_size: int = 100):
        """Initialize the ABO dataset handler.
        
        Args:
            abo_base_dir: Base directory containing ABO dataset
            subset_size: Number of images to include in the subset
        """
        self.abo_base_dir = Path(abo_base_dir)
        self.listings_dir = self.abo_base_dir / "abo-listings/listings/metadata"
        self.images_dir = self.abo_base_dir / "abo-images-small/images/small"
        self.subset_size = subset_size
        self.image_metadata = {}
        self.listings_data = []
        self.subset = []
        
    def load_image_metadata(self):
        """Load image metadata from CSV file."""
        images_csv_path = self.abo_base_dir / "abo-images-small/images/metadata/images.csv"
        
        if not os.path.exists(images_csv_path):
            print(f"Image metadata file not found: {images_csv_path}")
            return
            
        with open(images_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.image_metadata[row['image_id']] = {
                    'height': int(row['height']),
                    'width': int(row['width']),
                    'path': row['path']
                }
        print(f"Loaded metadata for {len(self.image_metadata)} images")
    
    def load_listings_sample(self, num_files: int = 1):
        """Load a sample of product listings from JSON files.
        
        Args:
            num_files: Number of listing files to load (each file contains many listings)
        """
        listing_files = list(self.listings_dir.glob("listings_*.json"))
        
        if not listing_files:
            print(f"No listing files found in {self.listings_dir}")
            return
            
        # Take a random sample of listing files
        sample_files = random.sample(listing_files, min(num_files, len(listing_files)))
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            listing = json.loads(line.strip())
                            # Only keep listings with images
                            if 'main_image_id' in listing and listing['main_image_id'] in self.image_metadata:
                                self.listings_data.append(listing)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.listings_data)} listings from {len(sample_files)} files")
    
    def create_subset(self):
        """Create a subset of listings with valid images for VQA task."""
        valid_listings = []
        
        for listing in self.listings_data:
            image_id = listing.get('main_image_id')
            if not image_id or image_id not in self.image_metadata:
                continue
                
            image_path = self.image_metadata[image_id]['path']
            full_image_path = self.images_dir / image_path
            
            if os.path.exists(full_image_path):
                valid_listings.append({
                    'listing': listing,
                    'image_path': str(full_image_path)
                })
        
        # Select a random subset of the desired size
        self.subset = random.sample(
            valid_listings, 
            min(self.subset_size, len(valid_listings))
        )
        
        print(f"Created subset with {len(self.subset)} items")
        return self.subset

class GeminiVQAGenerator:
    def __init__(self, api_key: str):
        """Initialize the Gemini VQA generator.
        
        Args:
            api_key: Google API key for accessing Gemini 2.0
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API requests.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_vqa_pairs(self, item: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, str]]:
        """Generate VQA pairs for an item using Gemini.
        
        Args:
            item: Dictionary containing listing data and image path
            num_questions: Number of questions to generate per image
            
        Returns:
            List of dictionaries with questions and answers
        """
        listing = item['listing']
        image_path = item['image_path']
        
        # Extract product metadata for context
        product_name = ""
        if 'item_name' in listing and listing['item_name']:
            for name_obj in listing['item_name']:
                if name_obj.get('language_tag') == 'en_US':
                    product_name = name_obj.get('value', '')
                    break
        
        product_description = ""
        if 'product_description' in listing and listing['product_description']:
            for desc_obj in listing['product_description']:
                if desc_obj.get('language_tag') == 'en_US':
                    product_description = desc_obj.get('value', '')
                    break
        
        # Include other potentially useful attributes
        color_info = ""
        if 'color' in listing and listing['color']:
            for color_obj in listing['color']:
                if color_obj.get('language_tag') == 'en_US':
                    color_info = color_obj.get('value', '')
                    break
        
        material_info = ""
        if 'material' in listing and listing['material']:
            for material_obj in listing['material']:
                if material_obj.get('language_tag') == 'en_US':
                    material_info = material_obj.get('value', '')
                    break
        
        # Load the image
        try:
            image = Image.open(image_path)
            
            # Prepare prompt for Gemini
            prompt = f"""
            Generate {num_questions} diverse visual question-answer pairs for this product image.
            
            Product information:
            - Name: {product_name}
            - Color: {color_info}
            - Material: {material_info}
            - Description: {product_description}
            
            Guidelines:
            1. Each question should be answerable by looking at the image.
            2. IMPORTANT: Answers MUST be exactly ONE WORD only. No phrases or multiple words allowed.
            3. Include a variety of question types: {', '.join(QUESTION_TYPES)}
            4. Include different difficulty levels: {', '.join(DIFFICULTY_LEVELS)}
            5. Questions should focus on visual attributes like color, shape, material, pattern, count, etc.
            6. For numbers, use digits (e.g., "3" instead of "three").
            7. Format all answers as a single word. For example:
               - "Red" (not "The product is red" or "red color")
               - "Square" (not "It is square" or "square shaped")
               - "3" (not "three" or "3 buttons")
            
            Format each Q&A pair exactly like this example:
            Question: What color is the cup?
            Answer: Red
            
            Only return the questions and answers in the specified format, nothing else.
            """
            
            # Make API call to Gemini
            response = self.model.generate_content([prompt, image])
            
            # Parse the response
            vqa_pairs = []
            if response and response.text:
                lines = response.text.strip().split('\n')
                i = 0
                while i < len(lines):
                    if i + 1 < len(lines) and lines[i].startswith("Question:") and lines[i+1].startswith("Answer:"):
                        question = lines[i].replace("Question:", "").strip()
                        answer = lines[i+1].replace("Answer:", "").strip()
                        
                        # Ensure answer is a single word
                        if len(answer.split()) == 1:
                            vqa_pairs.append({
                                "question": question,
                                "answer": answer
                            })
                        # Otherwise, try to extract the first word
                        else:
                            first_word = answer.split()[0]
                            # Remove any punctuation from the first word
                            first_word = first_word.strip('.,;:!?')
                            vqa_pairs.append({
                                "question": question,
                                "answer": first_word
                            })
                        i += 2
                    else:
                        i += 1
            
            return vqa_pairs
            
        except Exception as e:
            print(f"Error generating VQA pairs for {image_path}: {e}")
            return []

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate VQA pairs from ABO dataset using Gemini 2.0')
    parser.add_argument('--abo_dir', type=str, default='.', help='Base directory containing ABO dataset')
    parser.add_argument('--api_key', type=str, required=True, help='Google API key for Gemini')
    parser.add_argument('--subset_size', type=int, default=5000, help='Number of images to include in subset')
    parser.add_argument('--num_questions', type=int, default=10, help='Number of questions per image')
    parser.add_argument('--output_file', type=str, default='abo_vqa_dataset.csv', help='Output CSV file')
    parser.add_argument('--json_output', type=str, help='Optional JSON output file')
    args = parser.parse_args()
    
    # Initialize the dataset
    dataset = ABODataset(args.abo_dir, args.subset_size)
    
    # Load image metadata
    dataset.load_image_metadata()
    
    # Load a sample of product listings
    dataset.load_listings_sample(num_files=16)  # Start with 2 listing files
    
    # Create a subset for VQA
    subset = dataset.create_subset()
    
    if not subset:
        print("Failed to create subset. Exiting.")
        return
    
    # Initialize the Gemini VQA generator
    vqa_generator = GeminiVQAGenerator(args.api_key)
    
    # Generate VQA pairs
    vqa_dataset = []
    
    # Open CSV file for writing
    with open(args.output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['image_name', 'question', 'answer'])
        
        # Process each item
        for i, item in enumerate(subset):
            print(f"Processing item {i+1}/{len(subset)}")
            
            vqa_pairs = vqa_generator.generate_vqa_pairs(item, args.num_questions)
            
            if vqa_pairs:
                listing = item['listing']
                image_path = item['image_path']
                image_id = listing.get('main_image_id', '')
                image_filename = os.path.basename(image_path)
                
                # Extract product name
                product_name = ""
                if 'item_name' in listing and listing['item_name']:
                    for name_obj in listing['item_name']:
                        if name_obj.get('language_tag') == 'en_US':
                            product_name = name_obj.get('value', '')
                            break
                
                # Write each QA pair to CSV
                for qa_pair in vqa_pairs:
                    csv_writer.writerow([image_filename, qa_pair['question'], qa_pair['answer']])
                
                # Add to dataset for JSON output if requested
                if args.json_output:
                    vqa_dataset.append({
                        'image_id': image_id,
                        'image_path': image_path,
                        'product_name': product_name,
                        'vqa_pairs': vqa_pairs
                    })
    
    print(f"Generated VQA dataset saved to {args.output_file}")
    
    # Optionally save JSON format
    if args.json_output and vqa_dataset:
        with open(args.json_output, 'w') as f:
            json.dump(vqa_dataset, f, indent=2)
        print(f"JSON format also saved to {args.json_output}")

if __name__ == "__main__":
    main() 
