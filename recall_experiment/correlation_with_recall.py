import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl



causality = np.load("option.npy")

standardization_stats = []

causality_int = causality
causality_binary = np.zeros_like(causality_int) 

stage1 = causality_int[:, :]
for subj in range(stage1.shape[0]):
    for event in range(stage1.shape[1]):
        if stage1[subj, event] == 4:
            causality_binary[subj, event] = np.nan  
        else:
            causality_binary[subj, event] = np.where((stage1[subj, event] == 3),
                                                        1, 0)

causality_binary = causality_binary 

test=np.load("zhaiyao.npy")

event =np.zeros(50)

for subj in range(50):
    data=np.unique(test[subj])
    for i in range(1,51):
        if i in data:
            event[i-1]=event[i-1]+1


recall_counts=(event-np.mean(event))/np.std(event)
causality=np.nanmean(causality_binary[:,:],axis=0)
print(causality.shape)

data = pd.DataFrame({
    'Causality': causality,
    'RecallCounts': recall_counts
})


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['mathtext.fontset'] = 'stixsans'


v='Causality'
X = data[v] 
y = data['RecallCounts'] 
X
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

x = data[v].to_numpy()
y = data['RecallCounts'].to_numpy()
y_pred = model.predict(X)

mpl.rcParams['pdf.fonttype'] = 42


plot_df = pd.DataFrame({'x': X, 'y': y})
plot_df = plot_df.groupby(['x', 'y']).size().reset_index(name='count')

xu = plot_df['x'].to_numpy()
yu = plot_df['y'].to_numpy()
count = plot_df['count'].to_numpy()

x_range = xu.max() - xu.min() if len(xu) > 0 else 1.0
y_range = yu.max() - yu.min() if len(yu) > 0 else 1.0
if x_range == 0:
    x_range = 1.0
if y_range == 0:
    y_range = 1.0

plt.figure(figsize=(10, 6))


dx = 0.008 * x_range
dy = 0.012 * y_range

sizes_outer = np.full_like(count, 160, dtype=float)
sizes_mid   = np.full_like(count, 110, dtype=float)
sizes_inner = np.full_like(count, 60, dtype=float)
sizes_shadow = np.full_like(count, 170, dtype=float)

order = np.argsort(x)
x_line = x[order]
y_line = y_pred[order]

line_dx = 0.004 * x_range
line_dy = 0.006 * y_range


plt.plot(
    x_line + line_dx,
    y_line - line_dy,
    color='k',
    linewidth=5.2,
    alpha=0.12,
    solid_capstyle='round',
    zorder=1
)

plt.plot(
    x_line,
    y_line,
    color='#E67E22',
    linewidth=4.2,
    solid_capstyle='round',
    zorder=2,
    label='Regression Line'
)

plt.plot(
    x_line,
    y_line,
    color='#F28E2B',
    linewidth=2.1,
    alpha=0.95,
    solid_capstyle='round',
    zorder=3
)


plt.scatter(
    xu + dx, yu - dy,
    s=sizes_shadow,
    c='k',
    alpha=0.10,
    linewidths=0,
    zorder=4
)

plt.scatter(
    xu, yu,
    s=sizes_outer,
    c='#2C6DB2',
    edgecolors='#1E4F85',
    linewidths=0.9,
    alpha=0.98,
    zorder=6,
    label='Data Points'
)

plt.scatter(
    xu, yu,
    s=sizes_mid,
    c='#5FA8FF',
    linewidths=0,
    alpha=0.95,
    zorder=7
)

plt.scatter(
    xu, yu,
    s=sizes_inner,
    c='#B9DCFF',
    linewidths=0,
    alpha=0.92,
    zorder=8
)


plt.tick_params(axis='both', labelsize=18)
adjusted_r2 = model.rsquared_adj
p_value = model.pvalues[1]
correlation = X.corr(y)

textstr = f'Adjusted R² = {adjusted_r2:.3f}\np = {p_value:.3f}\nr={correlation:.3f}'
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()


plt.text(x_max - 0.32 * (x_max - x_min), y_min + 0.02 * (y_max - y_min), textstr, fontsize=18, bbox=dict(facecolor='white', alpha=0.0, edgecolor="none"))
plt.savefig('yin_correlation.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("已完成")
