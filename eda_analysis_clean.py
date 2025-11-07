import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_style('whitegrid')

DATA_DIR = '/scratch/zt1/project/msml612-fa25/user/harsh07/MedTeller'
REPORTS_CSV = f'{DATA_DIR}/data/indiana_reports.csv'
PROJECTIONS_CSV = f'{DATA_DIR}/data/indiana_projections.csv'

reports_df = pd.read_csv(REPORTS_CSV)
projections_df = pd.read_csv(PROJECTIONS_CSV)

print("="*70)
print("MEDTELLER: DATA EXPLORATION & EDA")
print("="*70)

print("\nBASIC STATISTICS:")
print(f"Total Reports: {len(reports_df):,}")
print(f"Total Images: {len(projections_df):,}")
print(f"Images per report: {len(projections_df)/len(reports_df):.2f}")

reports_df['findings_len'] = reports_df['findings'].fillna('').str.len()
reports_df['impression_len'] = reports_df['impression'].fillna('').str.len()

print("\nTEXT STATISTICS:")
print(f"Findings - Mean: {reports_df['findings_len'].mean():.0f}, Median: {reports_df['findings_len'].median():.0f}")
print(f"Impression - Mean: {reports_df['impression_len'].mean():.0f}, Median: {reports_df['impression_len'].median():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(reports_df['findings_len'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Findings Text Length', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')

axes[1].hist(reports_df['impression_len'], bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[1].set_title('Impression Text Length', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('text_lengths.png', dpi=300, bbox_inches='tight')
print("Saved: text_lengths.png")

all_problems = []
for problems in reports_df['Problems'].fillna('').str.split(';'):
    all_problems.extend([p.strip() for p in problems if p.strip()])

problem_counts = Counter(all_problems)
top_20 = problem_counts.most_common(20)

print("\nMEDICAL FINDINGS:")
print(f"Unique findings: {len(problem_counts)}")
print(f"Normal: {problem_counts.get('normal', 0):,} ({100*problem_counts.get('normal', 0)/len(reports_df):.1f}%)")

print("\nTop 10:")
for i, (problem, count) in enumerate(top_20[:10], 1):
    print(f"{i:2d}. {problem:30s} {count:5,} ({100*count/len(reports_df):5.1f}%)")

problems, counts = zip(*top_20)
plt.figure(figsize=(14, 8))
plt.barh(range(len(problems)), counts, color='steelblue')
plt.yticks(range(len(problems)), problems)
plt.xlabel('Frequency', fontsize=12)
plt.title('Top 20 Medical Findings in IU X-Ray Dataset', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top_findings.png', dpi=300, bbox_inches='tight')
print("Saved: top_findings.png")

projection_counts = projections_df['projection'].value_counts()

print("\nPROJECTIONS:")
for proj, count in projection_counts.items():
    print(f"  {proj:15s} {count:5,} ({100*count/len(projections_df):5.1f}%)")

images_per_report = projections_df.groupby('uid').size()
print(f"\nImages/report - Mean: {images_per_report.mean():.2f}, Median: {images_per_report.median():.0f}")

print("\n" + "="*70)
print("EDA COMPLETE - Dataset Ready for Model Development!")
print("="*70)

