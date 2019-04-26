import os
import re
import numpy as np
import matplotlib

matplotlib.use('Agg')

import pylab as plt

import datetime

ignore = ['ando', '.sh', '.png', '.py', '.git', 'README', 'data', 'script', 'test', '.idea']
maxcounts = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

user = list()
user.append("")
progress = list()
progress.append(np.array([0] * 15, dtype=np.float32))
for name in sorted([f for f in os.listdir() if os.path.isdir(f)]):
#for name in [f for f in os.listdir() if os.path.isdir(f)]:
    if any(name in igword for igword in ignore):
        continue
    user.append(name)
    print(name)
    score = list()
    for num, maxcount in zip(range(15), maxcounts):
        count = 0
        chapter = '{0:02d}'.format(num)
        if any(chapter in dirname for dirname in os.listdir(name)):
            #for script in os.listdir(os.path.join(name, chapter)):
            #    count += 1 if re.match(r'.+\.py', script) else 0
            count += 1
        score.append(min([count / maxcount, 1.0]))

    progress.append(np.array(score, dtype=np.float32))

user.append("")
progress.append(np.array([0] * 15, dtype=np.float32))
npscore = np.vstack(progress)
colors = ['orange', 'yellow', 'lime', 'green', 'turquoise', 'blue', 'indigo', 'deepskyblue','purple', 'pink', 'hotpink', 'sienna', 'darksalmon', 'grey', 'crimson']
labels = ['tutorial{0:02d}'.format(num) for num in range(15)]

offset = np.zeros(len(user))
for i in range(npscore.shape[1]):
    plt.bar(range(npscore.shape[0]), npscore[:, i], 0.6, offset, align='center', color=colors[i], label=labels[i])
    offset += npscore[:, i]


today = datetime.date.today()
#today = datetime.date(2017, 8, 2)
date_list = list()
#date_list.append(datetime.date(2019, 4, 19))
date_list.append(datetime.date(2019, 4, 26))
date_list.append(datetime.date(2019, 5, 10))
date_list.append(datetime.date(2019, 5, 17))
date_list.append(datetime.date(2019, 5, 24))
date_list.append(datetime.date(2019, 5, 31))
date_list.append(datetime.date(2019, 6, 7))
date_list.append(datetime.date(2019, 6, 14))
date_list.append(datetime.date(2019, 6, 21))
date_list.append(datetime.date(2019, 6, 28))
date_list.append(datetime.date(2019, 7, 5))
date_list.append(datetime.date(2019, 7, 12))
date_list.append(datetime.date(2019, 7, 19))
date_list.append(datetime.date(2019, 7, 26))
date_list.append(datetime.date(2019, 8, 2))


for i, d in enumerate(date_list):
    if today >= d:
        line = np.array([i + 1] * len(user), dtype=np.int32)
        label = "{}Border".format(str(d)[5:])
    elif today < date_list[0]:
        line = np.array([0] * len(user), dtype=np.int32)
        label = "StartLine"
        break
plt.plot(np.arange(0, len(user), 1), line, linewidth=4, color="red", label=label)

plt.xticks(range(npscore.shape[0]), user, fontsize=7, rotation = 45)
plt.yticks(np.arange(0, 16, 1))
plt.xlim(0, len(user) - 1)
plt.ylim(0, 15)
#plt.tight_layout()
plt.legend(fontsize=8, bbox_to_anchor=(1.27, 1.0))
plt.subplots_adjust(right=0.8)
plt.grid(axis='y', linestyle='dashed')
plt.savefig('progress.png')
plt.show()
