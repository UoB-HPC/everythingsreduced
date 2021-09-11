import pandas as pd
import glob
import csv

files = [
    "a100-results.csv",
    "clx-1S-results.csv",
    "clx-results.csv",
    "gen9-results.csv",
    "mi100-results.csv",
#    "rome-results-aocc.csv",
    "rome-results-cce.csv"]

csv_frames = []
for f in files:
    csv_frames.append(pd.read_csv(f, skipinitialspace=True))


df = pd.concat(csv_frames, axis=0, ignore_index=True)

df.loc[df['model'] == 'kokkos-sycl',['model']] = 'kokkos'

df.set_index(["kernel", "model", "arch", "compiler"], inplace=True)
df.sort_index(inplace=True)

avg = df.groupby(level=["kernel", "model", "arch", "compiler"]).mean()



peaks = pd.read_csv("peaks.csv", skipinitialspace=True)
peaks= pd.Series(peaks.bandwidth.values, index=peaks.arch).to_dict()

peakmap= {'rome': (2, 'EPYC 7742'),
          'clx_1S': (1, 'Xeon 6230'),
          'clx': (2, 'Xeon 6230'),
          'gen9': (1, 'Core 6770HQ')
          }

arches = avg.index.unique(level='arch')
for arch in arches:
    try:
        mul, key = peakmap[arch]
    except KeyError:
        mul, key = 1, arch
    avg.loc[(slice(None), slice(None), arch), 'bandwidth'] /= (mul*peaks[key])


app_name_map = {
        "openmp": "OpenMP",
        "kokkos-sycl" : "Kokkos (SYCL)",
        "omp-target": "OpenMP (target)",
        "onedpl": "oneDPL",
        "raja": "Raja",
        "kokkos": "Kokkos",
        "sycl": "SYCL",
    }
app_order = ['openmp',  'kokkos', 'raja', 'sycl', 'onedpl']

subapp_map = {
        'openmp' : 'openmp',
        'omp-target' : 'openmp',
        'kokkos' : 'kokkos',
        'kokkos-sycl' : 'kokkos',
        'raja' : 'raja',
        'sycl' : 'sycl',
        'onedpl' : 'onedpl',
    }


platform_name_map = {
        'clx' : "2 x Intel® Xeon® Gold 6230",
        'clx_1S' : "1 x Intel® Xeon® Gold 6230",
        'a100' : "NVIDIA A100",
        'mi100' : "AMD MI100",
        'rome' : '2 x AMD EPYC 7742',
        'rome_cce' : '2 x AMD EPYC 7742',
        'rome_aocc' : '2 x AMD EPYC 7742',
        'gen9' : 'Intel® Iris® Pro 580'
    }

for kernel in avg.index.unique(level='kernel'):
    with open(f"{kernel}.csv", "w") as fp:
        ocsv = csv.writer(fp)

        kslice = avg.loc[kernel]

        kslice.index.remove_unused_levels()
        models = kslice.index.unique(level='model')
        ocsv.writerow(["Device"] + list([app_name_map[x] for x in  models]))
        for arch in arches:
            res = [platform_name_map[arch]]
            for m in models:
                try:
                    v = avg.loc[(kernel, m, arch),'bandwidth'][0]*100
                except KeyError:
                    v = 'X'
                res.append(v)
            ocsv.writerow(res)
