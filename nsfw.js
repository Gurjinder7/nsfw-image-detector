import { pipeline, env} from '@huggingface/transformers'

export class MyNSFWDetectorPipeline {
    static task = 'image-classification';
    static model = 'AdamCodd/vit-base-nsfw-detector';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if(this.instance === null) {
            this.instance = pipeline(this.task, this.model, {progress_callback})
        }

        return this.instance
    }
}