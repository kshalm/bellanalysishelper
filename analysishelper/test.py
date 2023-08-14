import processBellData as pdb
import numpy as np
# import analysishelper.coinclib as cl
import coinclib as cl


def sanity_check(fname):
  '''Constructs PEFS from a given alice raw file, assuming
  bob and config files exist, and then check to see if we can certify entropy
  with that. Useful reference for data types and formats etc.'''
  alice_f = fname
  bob_f = fname.replace('_alice_', '_bob_')
  config_f = fname.replace('_alice_', '_config_').replace('.dat', '.yaml')

  f_dict = {'alice': alice_f,
            'bob': bob_f,
            'config': config_f,
            'output': 'save_compressed.dat'}

  freq, compressed, errors = pdb.process_single_run(
      f_dict, aggregate=True, findSync=True)
  return freq, errors


def fix_freqs_ch(freq):
  freqCH = np.zeros((4, 4))
  freqCH[0, :] = freq[3, :]
  freqCH[1, :] = freq[2, :]
  freqCH[2, :] = freq[1, :]
  # freqCH[1:3,:] = freq[1:3,:]
  freqCH[3, :] = freq[0, :] 
  # freqCH = fix_freqs(freqCH)
  return freqCH


path_to_data = '/Users/lks/Downloads_old/bell_data_test_errors/'
file_name_good = 'good/2023_04_24_23_12_alice_bafyriqhq5rxoplgbwrecxvq5ly2dym6tm2ta2z5v3cvwyygix5kd33eq3q73arsp7w4dzhiuvxcg4hpfnnv4neqaspphxgvbr6zse3n3zrabk_overnight_test_14_60s.dat'
file_name_bad = 'bad/2023_05_15_16_42_alice_bafyriqfjrtqo32pxep2rqmroixv6c2k2jdtan6xddlgb3dx6dtza5q6ahhdpnz7vrwvxh352sknrrigfh3c4xeuksumktxgyfmguulijhgldo_overnight_test_429_60s.dat'
file_name_bad2 = 'bad/2023_05_10_23_38_alice_bafyriqcjp2i45rdberpv6u7m22mocml27asdhrc24cobeztq6tfz4msn4balz6lvb6hziu6lrhrcp6rq5macma5v5golg3riahqdvywvzg6qo_overnight_test_323_60s.dat'
file_name_bad3 = 'bad/2023_07_20_00_07_alice_bafyriqa2lomjkc63i35phythrvx5wurt6oymsh3h47frsbjvsggemnhljsnkzu3qpsks26bt2ozhurrpzp3rmexqn3bvnmwhqhjazbcjehwim_production_run_3_56_60s.dat'
fname = path_to_data + file_name_good

freq, errors = sanity_check(fname)
# print(freq)
print('errors:', errors)

print('')
freqCH = fix_freqs_ch(freq).astype(int)
print(freqCH)
CH, CHn, ratio, pValue = cl.calc_violation(freqCH)
print(CH, ratio, pValue)
