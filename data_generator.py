import random
import csv

#identify shifts
shifts = [
    'First 6:00AM-2:30PM', 
    'Second 1:30PM to 9:30PM'
]

#identify materials
materials = [
    'Steel',
    '303 Stainless Steel',
    '304 Stainless Steel',
    '430 Stainless Steel',
    '6061 Aluminum',
    '6063 Aluminum',
    '7075 Aluminum'
]

#identify machine type
machine_type = ['Mill', 'Lathe']

#identify machines
lathe_machines = ['Miyano_BND', 'Big_Wia', 'Small_Wia', 'Samsung_SL15E']
mill_machines = ['YCM_1', 'YCM_2', 'YCM_3', 'Matsuura_MAM72', 'Doosan_HC400_II', 'NFX400A']

#identify customers 
customers = ['Tesla', 'Medtronic', 'Lockheed_Martin', 'Applied_Materials', 'General_Atomics']
industry_by_customer = {
    'Tesla': 'Automotive',
    'Medtronic': 'Medical',
    'Lockheed_Martin': 'Aerospace_Defense',
    'Applied_Materials': 'Semiconductor',
    'General_Atomics': 'Aerospace_Defense'
}

#identigy operators and experience level
operators = {
    'OP_001': 'Junior',
    'OP_002': 'Junior',
    'OP_003': 'Mid',
    'OP_004': 'Mid',
    'OP_005': 'Senior',
    'OP_006': 'Senior'
}

#identify tool conditions
tool_conditions = ['New', 'Mid_Life', 'Worn']

#identify workflow
inspection_types = ['First_Article', 'In_Process', 'Final_Inspection', '100%_Inspection']

#identify finish of part and difficulty
process_types = ['Standard', 'Tight_Tolerance', 'Secondary_Operation', 'Surface_Finish_Critical', 'High_Precision']

#identify defect types
defect_types = [
    'Dent',
    'Thread_Defect',
    'Incorrect_Hole_Size',
    'Incorrect_Depth',
    'Material_Cut_Too_Small',
    'Out_Of_Tolerance',
    'Damage_To_Surface',
    'Burrs_In_Part',
    'Missing_Feature',
    'Incorrect_Engraving',
    'Scratches'
]

#identify possible causes
possible_causes = [
    'Tool_On_Machine_Broke',
    'Incorrect_Print',
    'Incorrect_Revision_Level',
    'Operator_Mistake',
    'Loaded_Part_On_Wrong_Size',
    'Part_Lifted_On_Machine',
    'Careless_Deburring',
    'Programming_Error',
    'Machine_Broke_Down',
    'Incorrect_Tool_Used',
    'Material_Defect',
    'Measurement_Error',
    'Improper_Setup',
    'Damaged_When_Shipped_To_Customer'
]

#identify threads
has_threads = ['Yes', 'No']

#identify if deburr is needed
deburr = ['Yes', 'No']

#identify if outside process is needed
outside_process = ['Yes', 'No']

#identify if hardware is needed
hardware = ['Yes', 'No']

#identify outside process types
outside_process_types = [
    'Passivation',
    'Beadblast',
    'Grinding',
    'Clear_Chromate',
    'Clear_Hard_Anodize',
    'Chem_Film',
    'Colored_Hard_Anodize',
    'Gold_Anodize',
    'Heat_Treatment',
    'Nickel_Plating',
    'Electroless_Nickel_Plating',
    'Zinc_Plating',
    'White_Paint',
    'Colored_Paint'
]

#identify hardware types
hardware_type = ['Fasteners', 'Bolts', 'Screw', 'Dowel_Pins', 'Hinge']

#identify costs of scrap material
material_scrap_cost = {
    'Steel': 10,
    '303 Stainless Steel': 14,
    '304 Stainless Steel': 15,
    '430 Stainless Steel': 13,
    '6061 Aluminum': 9,
    '6063 Aluminum': 8,
    '7075 Aluminum': 16
}

def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def calculate_defect_rate(process, operator_level, tool_condition, rush_order,
                          machine_downtime, inspection_type, machine, material,
                          complexity_score, selected_deburr):
    rate = 0.008  # base defect rate = 0.8%

    # process difficulty
    if process == 'Tight_Tolerance':
        rate += 0.007
    elif process == 'Secondary_Operation':
        rate += 0.004
    elif process == 'Surface_Finish_Critical':
        rate += 0.006
    elif process == 'High_Precision':
        rate += 0.010

    # operator experience
    if operator_level == 'Junior':
        rate += 0.007
    elif operator_level == 'Mid':
        rate += 0.003
    elif operator_level == 'Senior':
        rate -= 0.002

    # tool condition
    if tool_condition == 'Mid_Life':
        rate += 0.003
    elif tool_condition == 'Worn':
        rate += 0.010

    # rush jobs and downtime
    if rush_order == 'Yes':
        rate += 0.006
    if machine_downtime == 'Yes':
        rate += 0.008

    # inspection catches more problems before shipment / final count
    if inspection_type == '100%_Inspection':
        rate -= 0.003
    elif inspection_type == 'Final_Inspection':
        rate -= 0.001

    # harder materials and jobs
    if selected_deburr == 'Manually Deburred':
        rate += 0.002
    if '7075' in material or '304' in material:
        rate += 0.002
    if machine == 'Small_Wia' and process in ['High_Precision', 'Tight_Tolerance']:
        rate += 0.003
    if machine == 'Matsuura_MAM72' and process == 'High_Precision':
        rate -= 0.002

    rate += (complexity_score - 5) * 0.0015

    return max(0.001, min(rate, 0.08))


def choose_defect(possible_defects_for_job, process, tool_condition, selected_deburr,
                  selected_thread, rush_order, machine_downtime):
    weights = []
    for defect in possible_defects_for_job:
        weight = 1.0

        if defect == 'Thread_Defect' and selected_thread == 'Yes':
            weight += 3
        if defect == 'Out_Of_Tolerance' and process in ['High_Precision', 'Tight_Tolerance']:
            weight += 3
        if defect == 'Incorrect_Hole_Size' and tool_condition == 'Worn':
            weight += 2
        if defect == 'Burrs_In_Part' and selected_deburr == 'Manually Deburred':
            weight += 2
        if defect == 'Scratches' and rush_order == 'Yes':
            weight += 2
        if defect == 'Damage_To_Surface' and rush_order == 'Yes':
            weight += 2
        if defect == 'Incorrect_Depth' and machine_downtime == 'Yes':
            weight += 2
        if defect == 'Missing_Feature' and process == 'Secondary_Operation':
            weight += 2

        weights.append(weight)

    return weighted_choice(possible_defects_for_job, weights)


def choose_cause(selected_defect, tool_condition, operator_level, machine_downtime, rush_order):
    if selected_defect == 'Thread_Defect':
        possible_causes_for_defect = [
            'Tool_On_Machine_Broke', 'Incorrect_Tool_Used', 'Programming_Error',
            'Improper_Setup', 'Measurement_Error'
        ]
    elif selected_defect == 'Scratches':
        possible_causes_for_defect = [
            'Careless_Deburring', 'Operator_Mistake', 'Damaged_When_Shipped_To_Customer'
        ]
    elif selected_defect == 'Incorrect_Hole_Size':
        possible_causes_for_defect = [
            'Incorrect_Tool_Used', 'Tool_On_Machine_Broke', 'Programming_Error', 'Improper_Setup'
        ]
    elif selected_defect == 'Burrs_In_Part':
        possible_causes_for_defect = [
            'Careless_Deburring', 'Tool_On_Machine_Broke', 'Incorrect_Tool_Used', 'Operator_Mistake'
        ]
    else:
        possible_causes_for_defect = possible_causes.copy()

    if tool_condition == 'Worn' and 'Tool_On_Machine_Broke' not in possible_causes_for_defect:
        possible_causes_for_defect.append('Tool_On_Machine_Broke')
    if operator_level == 'Junior' and 'Operator_Mistake' not in possible_causes_for_defect:
        possible_causes_for_defect.append('Operator_Mistake')
    if machine_downtime == 'Yes' and 'Machine_Broke_Down' not in possible_causes_for_defect:
        possible_causes_for_defect.append('Machine_Broke_Down')
    if rush_order == 'Yes' and 'Improper_Setup' not in possible_causes_for_defect:
        possible_causes_for_defect.append('Improper_Setup')

    return random.choice(possible_causes_for_defect)

records = []

for i in range(5000):
    job_id = f'JOB_{10000 + i}'

    selected_shift = random.choice(shifts)
    selected_customer = random.choice(customers)
    selected_industry = industry_by_customer[selected_customer]
    selected_material = random.choice(materials)
    selected_machine_type = random.choice(machine_type)

    if selected_machine_type == 'Lathe':
        selected_machine = random.choice(lathe_machines)
    else:
        selected_machine = random.choice(mill_machines)

    selected_operator = random.choice(list(operators.keys()))
    selected_operator_level = operators[selected_operator]

    selected_process = weighted_choice(
        process_types,
        [35, 22, 15, 12, 16]
    )

    if selected_process == 'Standard':
        complexity_score = random.randint(2, 4)
    elif selected_process == 'Secondary_Operation':
        complexity_score = random.randint(4, 6)
    elif selected_process == 'Surface_Finish_Critical':
        complexity_score = random.randint(5, 7)
    elif selected_process == 'Tight_Tolerance':
        complexity_score = random.randint(6, 8)
    else:
        complexity_score = random.randint(8, 10)

    selected_thread = weighted_choice(has_threads, [55, 45])

    if selected_thread == 'Yes':
        possible_defects_for_job = defect_types
    else:
        possible_defects_for_job = [
            'Dent', 'Incorrect_Hole_Size', 'Incorrect_Depth', 'Material_Cut_Too_Small',
            'Out_Of_Tolerance', 'Damage_To_Surface', 'Burrs_In_Part',
            'Missing_Feature', 'Incorrect_Engraving', 'Scratches'
        ]

    selected_deburr = random.choice(deburr)
    if selected_deburr == 'Yes':
        selected_deburr = 'Manually Deburred'
    else:
        selected_deburr = 'Part_Deburred_On_Machine'

    selected_tool_condition = weighted_choice(tool_conditions, [25, 50, 25])
    rush_order = weighted_choice(['Yes', 'No'], [18, 82])
    machine_downtime = weighted_choice(['Yes', 'No'], [10, 90])

    if complexity_score >= 8:
        inspection_type = weighted_choice(inspection_types, [10, 20, 25, 45])
    else:
        inspection_type = weighted_choice(inspection_types, [25, 35, 30, 10])

    selected_outside_process = random.choice(outside_process)
    if selected_outside_process == 'Yes':
        if 'Aluminum' in selected_material:
            possible_outside_process_for_job = outside_process_types.copy()
            for process_to_remove in ['Heat_Treatment', 'Grinding']:
                if process_to_remove in possible_outside_process_for_job:
                    possible_outside_process_for_job.remove(process_to_remove)
        else:
            possible_outside_process_for_job = outside_process_types.copy()
        selected_outside_process_type = random.choice(possible_outside_process_for_job)
    else:
        selected_outside_process_type = 'None'

    selected_hardware = random.choice(hardware)
    if selected_hardware == 'Yes':
        selected_hardware_type = random.choice(hardware_type)
    else:
        selected_hardware_type = 'None'

    # production volume shaped by complexity
    if complexity_score >= 8:
        total_parts = random.randint(10, 400)
    elif complexity_score >= 5:
        total_parts = random.randint(50, 1200)
    else:
        total_parts = random.randint(300, 3000)

    # setup and runtime
    base_setup = 20 + (complexity_score * 8)
    if selected_process == 'Secondary_Operation':
        base_setup += 15
    if selected_process == 'High_Precision':
        base_setup += 20
    if selected_operator_level == 'Junior':
        base_setup += 10
    elif selected_operator_level == 'Senior':
        base_setup -= 5
    if rush_order == 'Yes':
        base_setup -= 3
    setup_minutes = max(15, base_setup + random.randint(-5, 10))

    cycle_time_seconds = round(random.uniform(25, 70) + (complexity_score * random.uniform(4, 8)), 2)

    # weighted defect logic
    defect_percentage = calculate_defect_rate(
        selected_process,
        selected_operator_level,
        selected_tool_condition,
        rush_order,
        machine_downtime,
        inspection_type,
        selected_machine,
        selected_material,
        complexity_score,
        selected_deburr
    )

    defect_count = round(total_parts * defect_percentage)

    # costs
    machine_hourly_cost = 95 if selected_machine_type == 'Mill' else 80
    runtime_hours = (total_parts * cycle_time_seconds) / 3600
    machine_cost = round((setup_minutes / 60) * machine_hourly_cost + (runtime_hours * machine_hourly_cost), 2)

    deburr_cost = round(total_parts * 0.35, 2) if selected_deburr == 'Manually Deburred' else round(total_parts * 0.10, 2)
    outside_cost = round(total_parts * 0.60, 2) if selected_outside_process == 'Yes' else 0
    hardware_cost = round(total_parts * 0.20, 2) if selected_hardware == 'Yes' else 0

    scrap_cost_per_part = material_scrap_cost[selected_material] + (complexity_score * 1.5)
    scrap_cost = round(defect_count * scrap_cost_per_part, 2)

    production_cost = round(machine_cost + deburr_cost + outside_cost + hardware_cost + scrap_cost, 2)

    if defect_count > 0:
        selected_defect = choose_defect(
            possible_defects_for_job,
            selected_process,
            selected_tool_condition,
            selected_deburr,
            selected_thread,
            rush_order,
            machine_downtime
        )
        selected_cause = choose_cause(
            selected_defect,
            selected_tool_condition,
            selected_operator_level,
            machine_downtime,
            rush_order
        )
    else:
        selected_defect = 'None'
        selected_cause = 'None'

    record = {
        'Job ID': job_id,
        'Customer': selected_customer,
        'Industry': selected_industry,
        'Shift': selected_shift,
        'Operator ID': selected_operator,
        'Operator Level': selected_operator_level,
        'Material': selected_material,
        'Department': selected_machine_type,
        'Machine': selected_machine,
        'Process Type': selected_process,
        'Complexity Score': complexity_score,
        'Tool Condition': selected_tool_condition,
        'Rush Order': rush_order,
        'Machine Downtime': machine_downtime,
        'Inspection Type': inspection_type,
        'Outside Process': selected_outside_process,
        'Outside Process Type': selected_outside_process_type,
        'Hardware': selected_hardware,
        'Hardware Type': selected_hardware_type,
        'Deburr': selected_deburr,
        'Setup Minutes': setup_minutes,
        'Cycle Time Seconds': cycle_time_seconds,
        'Total Parts': total_parts,
        'Defect Count': defect_count,
        'Defect Rate': round(defect_percentage * 100, 2),
        'Thread': selected_thread,
        'Defect Type': selected_defect,
        'Possible Cause': selected_cause,
        'Scrap Cost Per Part': round(scrap_cost_per_part, 2),
        'Production Cost': production_cost
    }

    records.append(record)

# preview first 5 records
for record in records[:5]:
    print('\n------Part Summary-----')
    for key, value in record.items():
        print(f'{key}: {value}')

# write to csv
with open('manufacturing_defect_data.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print('\nCSV file created: manufacturing_defect_data.csv')
