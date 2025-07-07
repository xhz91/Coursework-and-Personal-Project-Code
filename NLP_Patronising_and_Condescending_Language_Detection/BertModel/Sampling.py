import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm

class DataSampling:
    def __init__(self,
                 fill_mask_model = "xlnet-base-cased", #"roberta-base",   # Model used to fill the mask space
                 paraphrasing_model = "t5-base",     # This have to be a t5 model
                 seed=42,
                 nltk_path="/vol/bitbucket/lg524/70016",
                 device=None):
        """
        Initialize the DataSampling class with required dependencies.
        Args:
            nltk_path (str, optional): Path to NLTK data directory
            seed (int, optional): Random seed for reproducibility
            device (str, optional): Device to use for model computations ('cuda' or 'cpu')
        """
        self.seed = seed
        # Initialize device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Setup fill mask pipeline
        self.fill_mask = pipeline("fill-mask", model=fill_mask_model)
        self.mask_token = self.fill_mask.tokenizer.mask_token
    
        # Setup T5 model for paraphrasing
        self.model_name = paraphrasing_model
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def downsample(self, raw_data):
        """
        Downsample the unpatronizing text data based on keywords to balance the dataset.   
        Args:
            raw_data (pd.DataFrame): Input dataframe with 'keyword' and 'label' columns
        Returns:
            pd.DataFrame: Downsampled and shuffled dataframe
        """
        keywords = raw_data['keyword'].unique()
        dfs = []
    
        for keyword in keywords:
            patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 1)]
            non_patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 0)]
            patro_count = len(patro_df)
            select_patro_df = non_patro_df.sample(n=patro_count*2, random_state=self.seed)
            downsampled_df = pd.concat([patro_df, select_patro_df])
            dfs.append(pd.concat([patro_df, downsampled_df]))
        
        result = pd.concat(dfs)
        return result.sample(frac=1, random_state=self.seed)

    def upsample(self, raw_data):
        """
        Upsample the patronizing text data based on keywords to balance the dataset.
        Args:
            raw_data (pd.DataFrame): Input dataframe with 'keyword' and 'label' columns
        Returns:
            pd.DataFrame: Upsampled and shuffled dataframe
        """
        dfs = []
    
        for keyword in raw_data['keyword'].unique():
            patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 1)]
            non_patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 0)]
            patro_count = len(patro_df)
            non_patro_count = len(non_patro_df)    

            n_complete_copies = non_patro_count // patro_count
            n_left_copies = non_patro_count % patro_count

            upsample = n_complete_copies * [patro_df]

            if n_left_copies > 0:
                n_left_samples = patro_df.sample(n=n_left_copies, replace=False, random_state=self.seed)
                upsample.append(n_left_samples)

            upsample_df = pd.concat(upsample)
            balanced_df = pd.concat([upsample_df, non_patro_df])
            dfs.append(balanced_df)
        
        result = pd.concat(dfs)
        return result.sample(frac=1, random_state=self.seed)
   

    def mask_and_fill(self, raw_data, mask_ratio=0.2):
        """
        Apply mask and fill augmentation to text data.    
        Args:
            raw_data (pd.DataFrame): Input dataframe with 'text' column
            mask_ratio (float): Ratio of tokens to mask in each text
        Returns:
            pd.DataFrame: Augmented dataframe with masked and filled text
        """
        result_df = raw_data.copy()

        for idx, row in result_df.iterrows():
            text = row['text']
            if len(text.split()) < 5:
                continue
        
            tokens = self.fill_mask.tokenizer(
             text, truncation=True, max_length=512, return_tensors="pt"
            )
            text = self.fill_mask.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            words = text.split()  

            num_mask = max(2, int(mask_ratio * len(words)))

            mask_indices = random.sample(range(len(words)), num_mask)

            for i in mask_indices:
                words[i] = self.mask_token
            masked_text = " ".join(words)
        
            result = self.fill_mask(masked_text)

            token_strs = [result[i][0]['token_str'] for i in range(len(result))]

            for i in range(num_mask):
                masked_text = masked_text.replace("<mask>", token_strs.pop(0), 1)
    
            result_df.at[idx, 'text'] = masked_text
    
        return result_df
   

    def upsampling_with_mask_and_fill(self, raw_data, mask_ratio=0.2):
        """
        Upsample the patronizing text data using mask and fill augmentation.
        Args:
            raw_data (pd.DataFrame): Input dataframe with 'keyword', 'label', and 'text' columns
            mask_ratio (float): Ratio of tokens to mask in each text
        Returns:
            pd.DataFrame: Upsampled and augmented dataframe
        """
        dfs = []
        for keyword in tqdm(raw_data['keyword'].unique(), desc = "Processing Keywords"):
            add_samples = []
            patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 1)]
            non_patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 0)]
            patro_count = len(patro_df)
            non_patro_count = len(non_patro_df)    
            n_complete_copies = non_patro_count // patro_count
            n_left_copies = non_patro_count % patro_count

            for _ in range(n_complete_copies-1):
                add_samples.append(self.mask_and_fill(patro_df, mask_ratio=mask_ratio))
            
            if n_left_copies > 0:
                n_left_samples = patro_df.sample(n=n_left_copies, replace=False, random_state=self.seed)
                add_samples.append(self.mask_and_fill(n_left_samples, mask_ratio=mask_ratio))
            
            add_samples.append(patro_df)
            upsample_df = pd.concat(add_samples)
            balanced_df = pd.concat([upsample_df, non_patro_df])
            dfs.append(balanced_df)
        
        result = pd.concat(dfs)
        return result.sample(frac=1, random_state=self.seed)

    def paraphrase_text(self, raw_data):
        """
        Paraphrase text using T5 model for data augmentation.

        Args:
            raw_data (pd.DataFrame): Input dataframe with 'text' column

        Returns:
            pd.DataFrame: Dataframe with original and paraphrased text
        """
        result_df = raw_data.copy()
        paraphrase_list = []

        for _, row in result_df.iterrows():
            new_row = row.copy()

            input_text = "paraphrase: " + row['text']
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

            output = self.model.generate(
                input_ids,
                max_length=200,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )

            paraphrase = self.tokenizer.decode(output[0], skip_special_tokens=True)

            if paraphrase == row['text'] or len(paraphrase) < 10:
                continue
            
            new_row['text'] = paraphrase
            paraphrase_list.append(new_row)
    
        paraphrase_df = pd.DataFrame(paraphrase_list)
        result_df = pd.concat([result_df, paraphrase_df], ignore_index=True)
        return result_df

    def upsampling_with_paraphrase_text(self, raw_data):
        """
        Upsample the patronizing text data using T5 paraphrasing.
    
        Args:
            raw_data (pd.DataFrame): Input dataframe with 'keyword', 'label', and 'text' columns
        
        Returns:
            pd.DataFrame: Upsampled and paraphrased dataframe
        """
        dfs = []
        keywords = raw_data[raw_data["label"] == 1]
        for keyword in tqdm(keywords['keyword'].unique(), desc = "Procesing Keywords"):
            add_samples = []
            patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 1)]
            non_patro_df = raw_data[(raw_data['keyword'] == keyword) & (raw_data['label'] == 0)]
            patro_count = len(patro_df)
            non_patro_count = len(non_patro_df)    

            n_complete_copies = non_patro_count // patro_count
            n_left_copies = non_patro_count % patro_count

            for _ in range(n_complete_copies-1):
                add_samples.append(self.paraphrase_text(patro_df))
            
            if n_left_copies > 0:
                n_left_samples = patro_df.sample(n=n_left_copies, replace=False, random_state=self.seed)
                add_samples.append(self.paraphrase_text(n_left_samples))
            
            add_samples.append(patro_df)
            upsample_df = pd.concat(add_samples)
            balanced_df = pd.concat([upsample_df, non_patro_df])
            dfs.append(balanced_df)
        
        result = pd.concat(dfs)
        return result.sample(frac=1, random_state=self.seed)