[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Yi0Zbe2y)
# MAST30034 Project 1 README.md
- Name: `QUYNH PHUONG LE`
- Student ID: `1288599`

**Research Goal:** My research goal is to predict trip duration for taxi trips given the pickup locations, dropoff locations, pickup and dropoff time. We will take into account several other factors affecting trip duration such as weather conditions and public holidays, etc.

**Timeline:** The timeline for the training set is Dec 2022 to May 2023. The timeline for the testing set is Jan 2024 to Mar 2024.
Please read the data dictionary for weather dataset at (https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/)
To run the pipeline, please first download the zone files (https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip) and placed the `taxi_zone' folder under the data/landing/ directory. Then run the files in order:
1. Run all files in 'scripts' folder.
2. Run 'Download and Filter.ipynb' to download taxi datasets and start basic filterings.
3. Run 'Preprocess weather.ipynb' to download weather datasets and preprocess them.
4. Run 'Visualisation.ipynb' to merge taxi and weather datasets, then visualise the data.
5. Run 'Modelling.ipynb' to fit models to each borough
