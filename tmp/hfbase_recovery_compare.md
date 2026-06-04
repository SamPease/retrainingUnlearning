# HFBase Recovery Comparison

Generated from local TOFU_SUMMARY artifacts.

## npo_forget10 on forget01
- taught
  - baseline: forget_quality=0.02369809422377763, model_utility=0.4322202580646987, forget_truth_ratio=0.6563694719625446
  - tuned: forget_quality=1.1983283569065546e-06, model_utility=0.5158727750274754, forget_truth_ratio=0.4633836281985978
  - retain_ref: forget_quality=5.354866135062879e-06, model_utility=0.4983537826060734, forget_truth_ratio=0.4734164434858513
- utility
  - baseline: retain90_utility=0.34860472135442616, retain_Q_A_Prob=0.42325927734375, retain_Q_A_ROUGE=0.2608011563943892, retain_Truth_Ratio=0.41514816388064885
  - tuned: retain90_utility=0.5280044507710923, retain_Q_A_Prob=0.63393798828125, retain_Q_A_ROUGE=0.47772374630208714, retain_Truth_Ratio=0.4972479933168044
  - retain_ref: retain90_utility=0.4971420467673388, retain_Q_A_Prob=0.56031494140625, retain_Q_A_ROUGE=0.4562008540864963, retain_Truth_Ratio=0.48596412112257936
- free_recovery
  - baseline: forget_quality=0.0003336198660466147, model_utility=0.4322202580646987, forget_truth_ratio=0.639165591535054
  - tuned: forget_quality=6.517010167622137e-10, model_utility=0.5158727750274754, forget_truth_ratio=0.5451291830882391
  - retain_ref: forget_quality=0.788243071988242, model_utility=0.4983537826060734, forget_truth_ratio=0.633270884015142

## npo_forget10 on forget05
- taught
  - baseline: forget_quality=0.00020094686161401939, model_utility=0.4322202580646987, forget_truth_ratio=0.6385877884845247
  - tuned: forget_quality=6.561871032713085e-17, model_utility=0.4550446529500326, forget_truth_ratio=0.4269696683311885
  - retain_ref: forget_quality=2.9793621192972874e-20, model_utility=0.3703398107198956, forget_truth_ratio=0.41424521948586424
- utility
  - baseline: retain90_utility=0.34860472135442616, retain_Q_A_Prob=0.42325927734375, retain_Q_A_ROUGE=0.2608011563943892, retain_Truth_Ratio=0.41514816388064885
  - tuned: retain90_utility=0.4546638698618763, retain_Q_A_Prob=0.452791748046875, retain_Q_A_ROUGE=0.4273202235434948, retain_Truth_Ratio=0.48790130455633585
  - retain_ref: retain90_utility=0.323449234140518, retain_Q_A_Prob=0.2246405029296875, retain_Q_A_ROUGE=0.3861417998954665, retain_Truth_Ratio=0.447678349759383
- free_recovery
  - baseline: forget_quality=0.017888195483849026, model_utility=0.4322202580646987, forget_truth_ratio=0.6433744014507045
  - tuned: forget_quality=3.2116698514542253e-06, model_utility=0.4550446529500326, forget_truth_ratio=0.5115639690496313
  - retain_ref: forget_quality=0.25438687552104255, model_utility=0.3703398107198956, forget_truth_ratio=0.5819639525920367

## npo_forget10 on forget10
- taught
  - baseline: forget_quality=0.0001305755477065129, model_utility=0.4322202580646987, forget_truth_ratio=0.6409680480063733
  - tuned: forget_quality=4.416671697087842e-24, model_utility=0.44960849766828515, forget_truth_ratio=0.4355181933916538
  - retain_ref: missing
- utility
  - baseline: retain90_utility=0.34860472135442616, retain_Q_A_Prob=0.42325927734375, retain_Q_A_ROUGE=0.2608011563943892, retain_Truth_Ratio=0.41514816388064885
  - tuned: retain90_utility=0.44294620807879265, retain_Q_A_Prob=0.415408935546875, retain_Q_A_ROUGE=0.41877181338725916, retain_Truth_Ratio=0.5056558479210083
  - retain_ref: missing

## rmu_forget10 on forget01
- taught
  - baseline: forget_quality=8.640363279162217e-06, model_utility=0.5882534063122767, forget_truth_ratio=0.4624121062499473
  - tuned: forget_quality=1.1983283569065546e-06, model_utility=0.5174716663846529, forget_truth_ratio=0.4489024737005158
  - retain_ref: forget_quality=5.354866135062879e-06, model_utility=0.4983537826060734, forget_truth_ratio=0.4734164434858513
- utility
  - baseline: retain90_utility=0.6558893194653193, retain_Q_A_Prob=0.82109375, retain_Q_A_ROUGE=0.6853056008046998, retain_Truth_Ratio=0.5271893404961238
  - tuned: retain90_utility=0.5198068075102099, retain_Q_A_Prob=0.5858544921875, retain_Q_A_ROUGE=0.466408075020413, retain_Truth_Ratio=0.5207190780380094
  - retain_ref: retain90_utility=0.4971420467673388, retain_Q_A_Prob=0.56031494140625, retain_Q_A_ROUGE=0.4562008540864963, retain_Truth_Ratio=0.48596412112257936
- free_recovery
  - baseline: forget_quality=6.788914643974486e-23, model_utility=0.5882534063122767, forget_truth_ratio=0.4671818201220083
  - tuned: forget_quality=3.436793945551114e-19, model_utility=0.5174716663846529, forget_truth_ratio=0.4557882377953634
  - retain_ref: forget_quality=0.788243071988242, model_utility=0.4983537826060734, forget_truth_ratio=0.633270884015142

## rmu_forget10 on forget05
- taught
  - baseline: forget_quality=8.871297583414431e-19, model_utility=0.5882534063122767, forget_truth_ratio=0.46151589708461543
  - tuned: forget_quality=1.6551869944948066e-19, model_utility=0.46019001546651145, forget_truth_ratio=0.4222840846378641
  - retain_ref: forget_quality=2.9793621192972874e-20, model_utility=0.3703398107198956, forget_truth_ratio=0.41424521948586424
- utility
  - baseline: retain90_utility=0.6558893194653193, retain_Q_A_Prob=0.82109375, retain_Q_A_ROUGE=0.6853056008046998, retain_Truth_Ratio=0.5271893404961238
  - tuned: retain90_utility=0.44594073932946104, retain_Q_A_Prob=0.42887451171875, retain_Q_A_ROUGE=0.41146140789539815, retain_Truth_Ratio=0.5088267950636634
  - retain_ref: retain90_utility=0.323449234140518, retain_Q_A_Prob=0.2246405029296875, retain_Q_A_ROUGE=0.3861417998954665, retain_Truth_Ratio=0.447678349759383
- free_recovery
  - baseline: forget_quality=4.214337327417989e-14, model_utility=0.5882534063122767, forget_truth_ratio=0.47209332148956207
  - tuned: forget_quality=1.7100063804653984e-13, model_utility=0.46019001546651145, forget_truth_ratio=0.4660284422442622
  - retain_ref: forget_quality=0.25438687552104255, model_utility=0.3703398107198956, forget_truth_ratio=0.5819639525920367

## rmu_forget10 on forget10
- taught
  - baseline: forget_quality=2.0132797133922014e-23, model_utility=0.5882534063122767, forget_truth_ratio=0.46669555830614756
  - tuned: forget_quality=4.353260441808186e-19, model_utility=0.43122769700628205, forget_truth_ratio=0.4449520528556708
  - retain_ref: missing
- utility
  - baseline: retain90_utility=0.6558893194653193, retain_Q_A_Prob=0.82109375, retain_Q_A_ROUGE=0.6853056008046998, retain_Truth_Ratio=0.5271893404961238
  - tuned: retain90_utility=0.4170170385582939, retain_Q_A_Prob=0.376798095703125, retain_Q_A_ROUGE=0.4012295321215472, retain_Truth_Ratio=0.4883598306756042
  - retain_ref: missing
