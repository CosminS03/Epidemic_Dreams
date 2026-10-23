FILE co_occurrence_graph.csv
FIELDS:
sym1 : name of the first medical condition (symptom)
sym2 : name of the second medical condition
count : number of times sym1 and sym2 co-occur in dream reports

FILES equally_prevalent.csv, more_prev_in_dreams_covid.csv, more_prev_in_twitter_covid.csv
FIELDS:
sym : name of the medical condition
rank_twitter_covid : rank of frequency of the medical condition in the twitter dataset (the lower the number, the more frequent the medical condition)
count_twitter_covid : frequency of the medical condition in the twitter dataset (raw count)
fq_twitter_covid : fraction of the frequency of this condition in twitter over the sum of all frequencies of all conditions in twitter
rank_dreams_covid : rank of frequency of the medical condition in the dream reports dataset (the lower the number, the more frequent the medical condition)
count_dreams_covid : frequency of the medical condition in the dream reports dataset (raw count)
fq_dreams_covid : fraction of the frequency of this condition in dream reports over the sum of all frequencies of all conditions in dream reports

FILE: only_in_dreams_covid.csv
FIELDS:
sym_only_in_dreams_covid : name of the medical condition
rank_dreams_covid : rank of frequency of the medical condition in the dream reports dataset (the lower the number, the more frequent the medical condition)
count_dreams_covid : frequency of the medical condition in the dream reports dataset (raw count)
fq_dreams_covid : fraction of the frequency of this condition in dream reports over the sum of all frequencies of all conditions in dream reports

FILE: only_in_twitter_covid.csv
FIELDS:
sym_only_in_twitter_covid : name of the medical condition
rank_twitter_covid : rank of frequency of the medical condition in the twitter dataset (the lower the number, the more frequent the medical condition)
count_twitter_covid : frequency of the medical condition in the twitter dataset (raw count)
fq_twitter_covid : fraction of the frequency of this condition in twitter over the sum of all frequencies of all conditions in twitter