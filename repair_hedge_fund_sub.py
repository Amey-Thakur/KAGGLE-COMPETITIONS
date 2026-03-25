import json
import os

def repair_hedge_fund_submission(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update Step 7 (Final Synthesis) to be more robust against column name expectations
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'Step 7: Final Synthesis' in ''.join(cell['source']):
            cell['source'] = [
                'print("Step 7: Final Synthesis & Format Review")\n',
                'if results_storage:\n',
                '    final_raw = pd.concat(results_storage)\n',
                '    \n',
                '    # Ensuring we match the EXACT test index and column naming the grader expects\n',
                '    # Some versions expect "id" + "target", others "id" + "y_target"\n',
                '    if "sample_sub" in locals():\n',
                '        submission = sample_sub[["id"]].merge(final_raw, on="id", how="left").fillna(0)\n',
                '    else:\n',
                '        submission = final_raw.copy()\n',
                '    \n',
                '    # Standardize output column to match the ground truth for "y_target"\n',
                '    if "prediction" in submission.columns:\n',
                '        submission = submission.rename(columns={"prediction": "target"})\n',
                '    \n',
                '    # Final strict column selection to avoid "Unexpected Error" from extra metadata\n',
                '    cols_to_save = ["id", "target"]\n',
                '    if "target" not in submission.columns and "y_target" in submission.columns:\n',
                '        submission = submission.rename(columns={"y_target": "target"})\n',
                '    \n',
                '    # Ensuring the ID is the first column and only target follows\n',
                '    final_sub = submission[cols_to_save].copy()\n',
                '    final_sub.to_csv("submission.csv", index=False)\n',
                '\n',
                '    print(f"File Exported Successfully. Rows: {len(final_sub)}")\n',
                '    print("Final Output Preview (Header Check):")\n',
                '    display(final_sub.head(5))\n',
                'else:\n',
                '    print("No inference results detected. Check Step 5.")\n'
            ]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    repair_hedge_fund_submission("Hedge Fund - Time Series Forecasting/Hedge_Fund_Time_Series_Forecasting.ipynb")
