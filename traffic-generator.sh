#!/bin/bash

while true
do

# LOW RISK
curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":0,
  "SeniorCitizen":0,
  "Partner":1,
  "Dependents":1,
  "tenure":72,
  "PhoneService":1,
  "MonthlyCharges":30,
  "TotalCharges":5000
}' > /dev/null


# HIGH RISK
curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":1,
  "SeniorCitizen":1,
  "Partner":0,
  "Dependents":0,
  "tenure":1,
  "PhoneService":1,
  "MonthlyCharges":120,
  "TotalCharges":120
}' > /dev/null


# FAILURE
curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":1,
  "SeniorCitizen":0,
  "Partner":1,
  "Dependents":0,
  "tenure":24,
  "PhoneService":1,
  "MonthlyCharges":-95,
  "TotalCharges":2300
}' > /dev/null

sleep 1

done
