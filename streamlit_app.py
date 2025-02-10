import numpy as np
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import pandas as pd
import plotly.graph_objects as go
from ast import literal_eval

# Material database (same as before)
materials_data = {
    # Ropes
    "Nylon Rope": {
        "type": "rope",
        "x": [1, 25, 49, 73, 97, 121],
        "y": [25.532, 657.819, 1395.374, 1073.109, 1152.845, 1259.159],
        "density": 1.14,
        "base_cost": 280,  # ₹ per kg
        "tensile_strength": 1200,
        "yield_strength": 800,
        "elongation_at_break": 30,
        "youngs_modulus": 3000,
        "standard": "ASTM D6267"
    },
    "Polypropylene Rope": {
        "type": "rope",
        "x": [1, 25, 49, 73, 97, 121],
        "y": [9.26964, 513.29844, 627.91848, 1051.51428, 742.53852, 822.2742],
        "density": 0.91,
        "base_cost": 220,
        "tensile_strength": 800,
        "yield_strength": 550,
        "elongation_at_break": 25,
        "youngs_modulus": 1500,
        "standard": "ISO 2307"
    },
    "Basalt Rope": {
        "type": "rope",
        "x": [1, 25, 49, 73, 97, 121],
        "y": [33.0818, 508.31496, 601.33992, 900.34872, 973.43976, 1073.1094],
        "density": 2.65,
        "base_cost": 450,
        "tensile_strength": 1500,
        "yield_strength": 1000,
        "elongation_at_break": 15,
        "youngs_modulus": 8000,
        "standard": "ASTM D7290"
    },
    # Rods
    "Nylon Rod": {
        "type": "rod",
        "x": [25, 25, 25, 25, 25, 49, 49, 49, 49, 49, 73, 73, 73, 73, 73, 97, 97, 97, 97, 97, 121, 121, 121, 121, 121],
        "y": [192.484, 176.338, 194.113, 187.645, 190.0645, 311.82408, 285.66756, 314.46306, 303.9849, 307.90449, 411.6077856, 415.0912392, 377.0811792, 401.260068, 406.4339268, 2525.510204, 2360.204082, 2106.122449, 2442.857143, 2233.163265, 2421.428571, 2591.836735, 2400, 2506.632653, 2495.918367],
        "density": 1.15,
        "base_cost": 340,
        "tensile_strength": 90,  # Example values - in MPa
        "youngs_modulus": 3000,  # MPa
        "yield_strength": 60,
        "elongation_at_break": 40,
        "standard": "ASTM D5947"
    },
    "Steel Rod": {
        "type": "rod",
        "x": [1, 5, 10, 15, 20, 25],
        "y": [785.6, 3928, 7856, 11784, 15712, 19640],
        "density": 7.85,
        "base_cost": 65,
        "tensile_strength": 400,  # MPa
        "yield_strength": 250,
        "elongation_at_break": 15,
        "youngs_modulus": 200000,  # MPa
        "standard": "ASTM A36"
    }
}

# Unit conversion factors (same as before)
unit_factors = {
    "kgf": {"kgf": 1, "N": 9.80665, "MPa": 0.1},
    "N": {"kgf": 0.101972, "N": 1, "MPa": 0.0101972},
    "MPa": {"kgf": 9.80665, "N": 98.0665, "MPa": 1},
    "g/cm^3": {"g/cm^3": 1, "kg/m^3": 1000},
    "kg/m^3": {"g/cm^3": 0.001, "kg/m^3": 1},
    "GPa": {"MPa": 1000, "GPa": 1},
    "MPa": {"MPa": 1, "GPa": 0.001},
}


def calculate_coefficients(x, y, degree=5):
    try:
        return np.polyfit(x, y, degree)
    except np.RankWarning:
        st.warning("Adjusting polynomial degree for better fit")
        return np.polyfit(x, y, min(degree, len(x) - 1))


def blend_materials(material1, material2, ratio):
    blended = {
        "type": material1["type"],
        "x": material1["x"],
        "y": [y1 * ratio + y2 * (1 - ratio) for y1, y2 in zip(material1["y"], material2["y"])],
        "density": material1["density"] * ratio + material2["density"] * (1 - ratio),
        "base_cost": material1["base_cost"] * ratio + material2["base_cost"] * (1 - ratio),
        "tensile_strength": material1["tensile_strength"] * ratio + material2["tensile_strength"] * (1 - ratio),
        "yield_strength": material1["yield_strength"] * ratio + material2["yield_strength"] * (1 - ratio),
        "elongation_at_break": material1["elongation_at_break"] * ratio + material2["elongation_at_break"] * (
                1 - ratio),
        "youngs_modulus": material1["youngs_modulus"] * ratio + material2["youngs_modulus"] * (1 - ratio),
        "standard": f"Blend of {material1['standard']} and {material2['standard']}"
    }
    return blended


def generate_pdf_report(data, results_table):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add custom style
    styles.add(ParagraphStyle(name='LeftAlign', alignment=0))

    # Title
    story.append(Paragraph("Textile Engineering Analyzer Report", styles['h1']))
    story.append(Spacer(1, 12))

    # Input Parameters (User's Choices)
    story.append(Paragraph("Input Parameters", styles['h2']))
    input_params_data = [
        ["Parameter", "Value"],
        ["Structure Type", data["Structure Type"]],
        ["Material", data["Material"]],
        ["Analysis Mode", "Strength Prediction" if "Element Count" in data else "Cost Optimization"],
        ["Strength Unit", data["Strength Unit"]],
        ["Density Unit", data["Density Unit"]],
        ["Young's Modulus Unit", data["Young's Modulus Unit"]],
    ]

    # Add optional parameters based on analysis mode
    if "Element Count" in data:  # Strength Prediction
        input_params_data.append(["Number of Elements", data["Element Count"]])
    else:  # Cost Optimization
        input_params_data.append(
            ["Target Strength", f"{data.get('Target Strength', 'N/A')} {data['Strength Unit']}"])  # Use .get()
        input_params_data.append(["Maximum Elements", data.get("Maximum Elements", "N/A")])

    input_params_table = Table(input_params_data)
    input_params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(input_params_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Material Properties", styles['h2']))
    material_props_data = [["Property", "Value"]]
    for k, v in data["Material Properties"].items():
        material_props_data.append([k, v])
    material_props_table = Table(material_props_data)
    material_props_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(material_props_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Results", styles['h2']))
    story.append(results_table)  # The results table passed to the function

    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(page_title="Textile Analyzer", layout="wide")
    st.title("Textile Engineering Analyzer")

    # About App Button and Modal
    if st.sidebar.button("About the App"):
        about_text = """
        ### Textile Engineering Analyzer: Application Overview

        This web application assists textile engineers and material scientists in analyzing and predicting the properties of ropes and rods, specifically their tensile strength. It allows users to:

        *   **Predict Tensile Strength:** Estimate the maximum load a rope or rod can withstand before breaking, based on the number of elements (yarns for ropes, individual rods for a rod structure) and the material properties.
        *   **Optimize for Cost:** Determine the minimum number of elements required to achieve a desired strength, while minimizing the overall cost of the material.
        *   **Analyze Predefined and Custom Materials:** Work with a built-in database of common textile materials or define custom blends by combining two materials in a specified ratio.
        *   **Upload material data:** Upload material data from a CSV file.
        *   **Generate Reports:** Create a PDF report summarizing the analysis parameters, results, and material properties.

        ### Key Terms and Properties

        *   **Tensile Strength:** The maximum stress a material can withstand before breaking when subjected to tension (being pulled apart). Measured in kgf, N, or MPa.
        *   **Yield Strength:** The stress at which a material begins to deform plastically (permanently). Measured in kgf, N, or MPa.
        *   **Elongation at Break:** The percentage increase in length that a material can withstand before breaking. A measure of ductility.
        *   **Young's Modulus:** A measure of a material's stiffness, or resistance to elastic deformation. Measured in MPa or GPa.
        *   **Density:** Mass per unit volume. Measured in g/cm³ or kg/m³.
        *   **Elements (Yarns/Rods):** The individual components that make up the textile structure. For ropes, these are yarns; for rod structures, these are individual rods.
        *   **Polynomial Regression:** A mathematical technique used to find the best-fitting curve (in this case, a polynomial) to a set of data points. This curve is used to predict the strength based on the number of elements.
        *   **Blending Ratio:** The proportion of each material used when creating a custom blend. A ratio of 0.5 means an equal mix of the two materials.

        ### Mathematical Steps

        #### Strength Prediction
        1.  **Data Input:** The user provides the number of elements and selects a material. The application retrieves the material's properties.
        2.  **Polynomial Regression:** The application uses polynomial regression to find the relationship between the number of elements (x) and tensile strength (y).
        3.  **Strength Prediction:** The polynomial equation is used to predict the tensile strength.
        4.  **Unit Conversion:** The predicted strength is converted to the user-selected unit.

        #### Cost Optimization

        1.  **Data Input:** The user provides the required strength and the maximum number of elements.
        2.  **Polynomial Regression:** Polynomial regression is used to find the relationship between elements and strength.
        3.  **Strength Calculation:** Tensile strength is calculated for a range of elements.
        4.  **Unit Conversion (Strength):** Strength values are converted to the user-selected unit.
        5.  **Feasible Solutions:** The application identifies element counts that meet or exceed the required strength.
        6.  **Optimal Solution:** The *minimum* number of elements that meets the requirement is selected.
        7.  **Cost Calculation:** Total cost is calculated (elements * density * cost per unit mass). Density is converted to g/cm³.
        8.  **Unit Conversion (Density):** Density is converted.

        ### Applications
        * Textile Design and Manufacturing
        * Material Selection
        * Quality Control
        * Education
        * Civil, Aerospace, and Marine Engineering

        ### Potential Improvements
        * Expanded Material Database
        * Additional Structure Types
        * More Detailed Analysis (e.g., FEA)
        * Environmental Factors
        * Fatigue Analysis
        * UI Enhancements
        * Database Integration
        * User Accounts
        * 3D Visualization
        * Mobile App

        ---
        *Made by Sakir and Jyot under the guidance of Dr. Jaita Sharma, Dr. Aadhar Mandot, and Mr. Ajay Pathak*
        """
        st.sidebar.markdown(about_text)

    # CSV Upload Section with Instructions and Example in a Collapsible Section
    st.subheader("Upload Material Data (CSV)")
    with st.expander("CSV File Format Help"):
        st.markdown("""
            Upload a CSV file containing material data.  The CSV file should have the following columns:

            *   **`Material Name`:**  (Text) The name of the material.
            *   **`Type`:** (Text)  Must be either "rope" or "rod" (case-insensitive).
            *   **`x`:** (Text) A Python list of numbers representing the number of elements (e.g., `"[1, 25, 49]"`).  **Important:** Use square brackets and commas.
            *   **`y`:** (Text) A Python list of numbers representing the corresponding tensile strengths (e.g., `"[25.5, 657.8, 1395.4]"`). **Important:** Use square brackets and commas.  `x` and `y` must have the same length.
            *   **`Density`:** (Number) The density of the material.
            *   **`Cost`:** (Number) The base cost per unit mass (₹/kg).
            *   **`Tensile Strength`:** (Number, optional)  Tensile strength of the material.
            *   **`Yield Strength`:** (Number, optional) Yield strength of the material.
            *   **`Elongation at Break`:** (Number, optional) Elongation at break (%).
            *   **`Young's Modulus`:** (Number, optional) Young's Modulus of the material.
            *   **`Standard`:** (Text, optional)  The relevant standard for the material.

            **Example CSV Data:**

            ```csv
            Material Name,Type,x,y,Density,Cost,Tensile Strength,Yield Strength,Elongation at Break,Young's Modulus,Standard
            My Custom Rope,rope,"[1, 10, 20, 30]","[10, 100, 220, 350]",1.2,150,1100,700,28,2800,ASTM XYZ
            Another Rod,rod,"[5, 10, 15]","[50, 110, 180]",7.5,80,450,280,12,210000,ASTM ABC
            ```
            **How to Convert Excel to CSV:**
            1.  Open your Excel file.
            2.  Go to "File" -> "Save As".
            3.  Choose "CSV (Comma delimited) (*.csv)" as the file type.
            4.  Click "Save".
        """)

    col1_upload, col2_upload = st.columns([3, 1])  # Adjust column widths as needed

    with col1_upload:
      uploaded_file = st.file_uploader("Upload CSV", type="csv", key="file_uploader")
    with col2_upload:
      st.markdown("") #for allign
      st.markdown("") #for allign
      if st.button("Help"):
          st.write("Click on 'CSV File Format Help' above to see the required format.")


    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.write(df_uploaded)  # Display the uploaded data

            # Convert uploaded data to the required format.
            new_materials_data = {}
            for index, row in df_uploaded.iterrows():
                # Basic error handling
                try:
                    x_values = literal_eval(row['x'])  # Use eval safely with literal_eval
                    y_values = literal_eval(row['y'])
                    if not (isinstance(x_values, list) and isinstance(y_values, list)):
                        raise ValueError("x and y must be lists")
                    if len(x_values) != len(y_values):
                        raise ValueError("x and y lists must have the same length")

                    new_materials_data[row['Material Name']] = {
                        "type": row['Type'].lower(),  # important to convert .lower()
                        "x": x_values,
                        "y": y_values,
                        "density": float(row['Density']),
                        "base_cost": float(row['Cost']),
                        "tensile_strength": float(row.get('Tensile Strength', 0)),  # Use .get() for optional columns
                        "yield_strength": float(row.get('Yield Strength', 0)),
                        "elongation_at_break": float(row.get('Elongation at Break', 0)),
                        "youngs_modulus": float(row.get("Young's Modulus", 0)),
                        "standard": row.get('Standard', 'N/A')
                    }
                except (ValueError, SyntaxError, TypeError) as e:
                    st.error(f"Error processing row {index + 1}: {e}.  Skipping this row.")
                    continue  # Skip to the next row

            # Merge with existing data (optional, if you want to keep the original data)
            materials_data.update(new_materials_data)
            st.success("CSV data loaded and merged successfully!")


        except Exception as e:
            st.error(f"Error reading or processing CSV file: {e}")

    with st.sidebar:
        st.header("Configuration")
        structure_type = st.radio("Structure Type:", ("Rope", "Rod"))

        # Filter materials based on selected type
        filtered_materials = {
            k: v for k, v in materials_data.items()
            if v["type"] == structure_type.lower()
        }

        material_choice = st.radio("Material Selection:", ("Predefined", "Custom Blend"))

        if material_choice == "Custom Blend":
            materials = list(filtered_materials.keys())
            mat1 = st.selectbox("Primary Material:", materials)
            mat2 = st.selectbox("Secondary Material:", materials)
            blend_ratio = st.slider("Blend Ratio", 0.0, 1.0, 0.5)
            selected_data = blend_materials(filtered_materials[mat1], filtered_materials[mat2], blend_ratio)
            selected_data["current_cost"] = st.number_input("Cost (₹/kg):",
                                                            value=float(selected_data["base_cost"]),
                                                            min_value=1.0,
                                                            step=10.0)
        else:
            selected_material = st.selectbox("Select Material:", list(filtered_materials.keys()))  # Convert to list
            selected_data = filtered_materials[selected_material].copy()
            selected_data["current_cost"] = st.number_input("Edit Cost (₹/kg):",
                                                            value=float(selected_data["base_cost"]),
                                                            min_value=1.0,
                                                            step=10.0)

        # Unit selection
        strength_unit = st.selectbox("Strength Unit:", ("kgf", "N", "MPa"))
        density_unit = st.selectbox("Density Unit:", ("g/cm^3", "kg/m^3"))
        youngs_modulus_unit = st.selectbox("Young's Modulus Unit", ("MPa", "GPa"))
        # ... (Add other unit selections as needed)

        analysis_mode = st.radio("Analysis Mode:", ("Strength Prediction", "Cost Optimization"))

    if analysis_mode == "Strength Prediction":
        col1, col2 = st.columns(2)
        with col1:
            element_count = st.number_input(
                f"Number of {structure_type}s:" if structure_type == "Rod" else "Number of Yarns:",
                min_value=1,
                value=50
            )

        with col2:
            coefficients = calculate_coefficients(selected_data["x"], selected_data["y"])

            # Convert predicted strength based on units
            if structure_type == "Rope":
                predicted_strength_raw = np.polyval(coefficients, element_count)
                predicted_strength = predicted_strength_raw * unit_factors[strength_unit]['kgf']
            else:
                predicted_strength_raw = np.polyval(coefficients, element_count)  # Rod strength in MPa
                predicted_strength = predicted_strength_raw * unit_factors[strength_unit]['MPa']

            st.metric(
                "Predicted Tensile Strength",
                f"{predicted_strength:.2f} {strength_unit}",
                help="Maximum load capacity before failure"
            )

        # Plotly chart
        fig = go.Figure()
        x_range = np.linspace(min(selected_data["x"]), max(selected_data["x"]) * 1.2, 100)
        y_pred = np.polyval(coefficients, x_range)

        # Convert y_pred for plotting if necessary.
        if structure_type == "Rope":
            y_pred_converted = y_pred * unit_factors[strength_unit]['kgf']
        else:
            y_pred_converted = y_pred * unit_factors[strength_unit]['MPa']

        fig.add_trace(go.Scatter(x=x_range, y=y_pred_converted, mode='lines', name='Predicted Strength'))
        fig.add_trace(go.Scatter(x=selected_data["x"],
                                 y=[y * unit_factors[strength_unit]['kgf'] if structure_type == 'Rope' else y *
                                    unit_factors[strength_unit]['MPa'] for y in selected_data['y']], mode='markers',
                                 name='Data Points'))  # Convert datapoints too
        fig.add_vline(x=element_count, line_dash="dash", line_color="green", annotation_text="Selected Count")
        fig.update_layout(
            xaxis_title="Number of Elements",
            yaxis_title=f"Tensile Strength ({strength_unit})",
            title="Strength Prediction",
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=100,
                r=20,
                t=110,
            ),

            plot_bgcolor='white'

        )
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_mode == "Cost Optimization":
        col1, col2 = st.columns([1, 2])
        with col1:
            target_strength = st.number_input(f"Required Strength ({strength_unit}):",
                                                min_value=1.0,
                                                value=500.0 if strength_unit != 'MPa' else 50.0)  # Default for MPa
            max_elements = st.number_input("Maximum Elements:",
                                            min_value=1,
                                            value=200)

        coefficients = calculate_coefficients(selected_data["x"], selected_data["y"])
        elements_range = np.arange(1, max_elements + 1)

        # Calculate strength values and convert them to the selected unit.
        if structure_type == "Rope":
            strength_values_raw = np.polyval(coefficients, elements_range)
            strength_values = strength_values_raw * unit_factors[strength_unit]['kgf']
        else:  # Rod
            strength_values_raw = np.polyval(coefficients, elements_range)
            strength_values = strength_values_raw * unit_factors[strength_unit]['MPa']

        valid_solutions = elements_range[strength_values >= target_strength]

        if len(valid_solutions) > 0:
            optimal_elements = valid_solutions[0]
            # Convert density to consistent units
            density = selected_data["density"] * unit_factors[density_unit]["g/cm^3"]  # Convert to g/cm^3
            if structure_type == 'Rope':
                material_mass = optimal_elements  # Number of yarns
            else:
                material_mass = optimal_elements  # Number of rods
            total_cost = material_mass * selected_data["current_cost"] * density  # Use g/cm3

            with col1:
                st.metric("Optimal Elements", optimal_elements)
                st.metric("Total Cost", f"₹{total_cost:,.2f}")

            with col2:
                df = pd.DataFrame({
                    "Elements": elements_range,
                    "Strength": strength_values,
                    "Cost": [e * selected_data["density"] * unit_factors[density_unit]["g/cm^3"] * selected_data[
                        "current_cost"] for e in elements_range]  # Consistent units.
                })
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Elements"], y=df["Strength"], mode='lines', name='Strength'))
                fig.add_trace(go.Scatter(x=df["Elements"], y=df["Cost"], mode='lines', name='Cost'))
                fig.add_vline(x=optimal_elements, line_dash="dash", line_color="green",
                              annotation_text=f"Optimal: {optimal_elements}")
                fig.update_layout(
                    xaxis_title="Number of Elements",
                    yaxis_title="Value",
                    title="Cost vs. Strength",
                    xaxis=dict(showline=True, showgrid=False, showticklabels=True, linewidth=2, ticks='outside'),
                    yaxis=dict(showline=True, showgrid=True, showticklabels=True, linewidth=2, ticks='outside'),
                    plot_bgcolor='white'
                )

                st.plotly_chart(fig, use_container_width=True)


        else:
            st.error("No feasible solution within given constraints")

    if st.button("Generate Technical Report"):
        report_data = {
            "Structure Type": structure_type,
            "Material": selected_material if material_choice == "Predefined" else f"{mat1} / {mat2}",
            "Element Count": element_count if analysis_mode == "Strength Prediction" else optimal_elements,
            "Unit Cost": f"₹{selected_data['current_cost']}/kg",
            "Total Estimated Cost": f"₹{total_cost:,.2f}" if analysis_mode == "Cost Optimization" else "N/A",
            "Strength Unit": strength_unit,
            "Density Unit": density_unit,
            "Young's Modulus Unit": youngs_modulus_unit,
            "Material Properties": {
                "Young's Modulus": f"{selected_data['youngs_modulus']:.2f} {youngs_modulus_unit}",
                "Tensile Strength": f"{selected_data['tensile_strength']:.2f} {strength_unit if strength_unit != 'MPa' else 'MPa'}",
                "Yield Strength": f"{selected_data['yield_strength']:.2f} {strength_unit if strength_unit != 'MPa' else 'MPa'}",
                "Elongation at Break": f"{selected_data['elongation_at_break']:.2f} %",
                "Standard": selected_data["standard"]
            },
            "Target Strength": target_strength if analysis_mode == "Cost Optimization" else None,  # Add target strength
            "Maximum Elements": max_elements if analysis_mode == "Cost Optimization" else None  # Add max elements

        }

        # Create results table for PDF
        if analysis_mode == "Cost Optimization":
            results_data = [["Elements", "Strength (" + strength_unit + ")", "Cost (₹)"]]
            for e, s, c in zip(elements_range, strength_values,
                               [e * selected_data["density"] * unit_factors[density_unit]["g/cm^3"] * selected_data[
                                   "current_cost"] for e in elements_range]):  # Consistent units
                results_data.append([e, f"{s:.2f}", f"{c:.2f}"])
        else:  # Strength prediction mode
            results_data = [["Number of Elements", "Predicted Strength (" + strength_unit + ")"]]
            results_data.append([element_count, f"{predicted_strength:.2f}"])

        results_table = Table(results_data)
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        pdf_buffer = generate_pdf_report(report_data, results_table)
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="textile_analysis_report.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
