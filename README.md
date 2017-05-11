# PatientKind-Diseases-Specialists-Matching

Rare diseases collectively affect over 350 million people worldwide, 50% of which
are children [1, 2]. Because each disease is rare, a general physician may have
never seen a patient with a particular disease in their entire career [3]. Symptoms
can also vary considerably between individual cases, making correct diagnosis
difficult [4]. Misdiagnosis and incorrect treatment are extremely costly and potentially
dangerous, so it is important to quickly refer patients to specialists that are experts
in their condition. To facilitate this referral process, we trained machine learning
models to predict the expertise of each researcher in each rare disease, based
on their publication record. We compared the performance of three methods on
a dataset of 209,110 disease-author associations, and were able to classify rare
disease experts from GeneReviews with 76% accuracy and predict 21,224 new
disease-expert associations.

Running the program, you can follow the order:\\
GeneReviews_Scraping.py -> GeneReviews_Preprocessing.py -> OMIM_Scraping.py -> training_data_generation.py -> Different Learning Algorithms
