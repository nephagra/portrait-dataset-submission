import json
with open('js/portraits.json', 'r') as f:
    metadata = json.load(f)

times = [ float(x['mid-date']) for x in metadata ]
fnames = [ x['filename'] for x in metadata ]

time_periods = {
    "medieval": (1000, 1399),
    "early-renaissance": (1400, 1494),
    "high-renaissance": (1495, 1519), 
    "late-Renaissance": (1520, 1590), 
    "baroque": (1590, 1700), 
    "neoclassicism": (1700, 1805), 
    "romanticism": (1805, 1850), 
    "realism": (1840, 1880), 
    "impressionism": (1870, 1890), 
    "post-impressionism": (1886, 1910), 
    "modernism": (1910, 1965),
    "post-modernism": (1965, 2019),
}


period_fnames = {
    k: [ fnames[i] for i, t in enumerate(times) if t > v[0] and t < v[1]  ]
    for k, v in time_periods.items()
}

print({ k: len(v) for k, v in period_fnames.items() })

with open('js/time_periods.json', 'w') as f:
    json.dump(period_fnames, f)


