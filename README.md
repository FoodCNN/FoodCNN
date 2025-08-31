# FoodCNN

Zero-shot nutrition estimation from a single food photo.  
Deep learning models to predict dish weight, calories, and macronutrients (fat, carbs, protein) using the Nutrition5K dataset.  

Authors: Georgy Salakhutdinov, Georgii Kuznetsov, Ayoub Agouzoul  
École Polytechnique (X), 2025

## Abstract

<p align="justify">
Keeping track of our food intake can still feel like a chore, even with so many nutrition apps available today. This project explores a potential solution through Machine Learning. We frame this challenge as a multi-output regression task. Given an RGB image of a plated dish, we want to predict its total weight together with calories, fat, carbs, and protein. The Nutrition5K dataset from Google Research has been utilized for training. This paper makes 3 contributions. Firstly, we present out data pipeline, which involves cleaning, segmentation, and a Principal Component Analysis (PCA) video frame extraction strategy. Secondly, we present our model WeightCNN that has been trained to predict weight with 18.7% relative MAE. Finally, we present a family of relative-macro models that have been trained to estimate macro percentages. These models are then combined with WeightCNN to obtain the full calorie and macronutrient prediction from a single photo.
</p>

## Resources
- [Slides (PDF)](Slides.pdf) – project presentation
- [Report (PDF)](report/FoodCNN_Report.pdf) – full technical report
