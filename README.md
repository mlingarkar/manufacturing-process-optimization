# Manufacturing Process Optimization

A Python project that simulates manufacturing job data and analyzes the tradeoff between **quality** and **cost** in a machine shop environment.

This project simulates manufacturing jobs across different machines, shifts, materials, process types, deburring methods, hardware requirements, and outside processing needs. It then evaluates machine performance using both **defect rate** and **production cost**, with a weighted optimization model that prioritizes quality over cost.

## Project Goals

- Minimize defect rate
- Minimize production cost
- Prioritize quality over cost using a **70/30 weighting model**
- Compare machine performance using both tables and visualizations

## Project Files

- `data_generator.py`  
  Generates simulated manufacturing job data and exports it to `manufacturing_defect_data.csv`

- `analysis.py`  
  Loads the generated dataset, calculates summary statistics, creates graphs, and computes an optimization score for each machine

- `manufacturing_defect_data.csv`  
  Simulated dataset containing manufacturing job records

## How the Project Works

### 1. Feature Data Generation
The project first generates simulated manufacturing jobs using:
- machine type
- shift
- material
- process type
- deburr method
- hardware requirements
- outside processing
- defect count
- defect rate
- estimated production cost

### 2. Data Analysis
The generated data is analyzed to measure:
- average defect rate by machine
- average defect rate by shift
- total defects by material
- most common defect types
- average production cost by machine
- combined machine performance across both cost and quality

### 3. Optimization
A weighted score is calculated for each machine using:
- **70% defect rate**
- **30% production cost**

This reflects a realistic machining priority where **quality matters more than cost**.

## Visualizations Included

This project includes graphs to help compare manufacturing performance, such as:
- average defect rate by machine
- average defect rate by shift
- total defects by material
- average production cost by machine

These visualizations make it easier to identify high-risk machines, costly production setups, and quality trends across the shop.

## Tools Used

- Python
- pandas
- matplotlib

## Key Results

The project helps identify:
- the machine with the lowest average defect rate
- the machine with the lowest average production cost
- the machine with the best overall weighted optimization score

Because quality is weighted more heavily than cost, the final recommendation favors machines that reduce defects even if they are not the absolute cheapest option.

## Why This Project Matters

This project demonstrates how data analysis and optimization can be applied to manufacturing decisions. Instead of only describing defect trends, it helps answer a more practical business question:

**Which machine provides the best balance between production quality and cost?**

## Future Improvements

- optimize by both **machine and shift**
- add more realistic machine-specific defect behavior
- include estimated cycle time or hourly machine rates
- build an interactive dashboard
- allow a user to input a job type and receive the recommended machine setup
