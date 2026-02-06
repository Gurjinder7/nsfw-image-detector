// import { MyNSFWDetectorPipeline } from "./nsfw.js";

import { pipeline, env} from '@huggingface/transformers'

class MyNSFWDetectorPipeline {
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

export const classifyImage = async (url) => {
    console.log(url)
      try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to load image');

    const blob = await response.blob();
    console.log(blob)
    const image = {};
    const imagePromise = new Promise((resolve, reject) => {
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = URL.createObjectURL(blob);
    });

    console.log(image)
    const classifier = await MyNSFWDetectorPipeline.getInstance()
    // const img = await imagePromise; // Ensure the image is loaded
    console.log(classifier)
    // console.log(img)
    const classificationResults = await classifier([image.src]); // Classify the image
    console.log('Predicted class: ', classificationResults[0]);
  } catch (error) {
    console.error('Error classifying image:', error);
  }
}


classifyImage('https://dbuzz-assets.s3.amazonaws.com/ai_image/public/fl/image-1770335175109.jpeg')