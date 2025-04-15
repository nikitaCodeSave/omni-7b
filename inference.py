from qwen_omni_utils import process_mm_info
import torch
import gc

# Helper function to clear GPU memory
def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# @title inference function
def inference(audio_path, prompt, sys_prompt, model, processor, use_audio_in_video=False, max_new_tokens=512):
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": audio_path},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process multimedia info
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        
        # Pre-process
        inputs = processor(
            text=text, 
            audios=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            use_audio_in_video=use_audio_in_video
        )
        
        # Ensure correct tensor types - categorical data should stay as Long
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        
        # Move to device without changing dtype for categorical data
        if input_ids is not None:
            inputs['input_ids'] = input_ids.to(model.device)
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask.to(model.device)
        
        # Handle other tensors - match the model's dtype for non-categorical data
        for k, v in inputs.items():
            if k not in ['input_ids', 'attention_mask'] and isinstance(v, torch.Tensor):
                # Apply model dtype only to float tensors
                if v.dtype in [torch.float, torch.float16, torch.float32, torch.bfloat16]:
                    inputs[k] = v.to(model.device, dtype=model.dtype)
                else:
                    inputs[k] = v.to(model.device)
        
        # Fix for mask shape mismatch
        if 'attention_mask' in inputs and 'input_ids' in inputs:
            if inputs['attention_mask'].shape[1] != inputs['input_ids'].shape[1]:
                seq_len = inputs['input_ids'].shape[1]
                inputs['attention_mask'] = inputs['attention_mask'][:, :seq_len]
        
        del audios, images, videos
        
        # Generate output with explicit dtype handling
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                use_audio_in_video=use_audio_in_video, 
                return_audio=False, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
        text = processor.batch_decode(output, skip_special_tokens=True)
        
        del inputs, output
        
        return text
    except Exception as e:
        print(f"Error in inference: {str(e)}")
        # Include stack trace for debugging
        import traceback
        traceback.print_exc()
        return [f"Error transcribing audio: {str(e)}"]