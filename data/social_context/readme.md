Put your Tweets soical context dataset in ```social_context``` directory for training and test but do not upload to git repository

The context directory ```social_context``` is expected to be the top directory of one or more subdirectory containing events dataset directories (inc. 'rumours' and 'non-rumours'), see format of PHEME dataset for details
and we are using latest PHEME 6392078 veracity dataset (available via 'https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078')

The recommended way is to create your local symlink for either ```all-rnr-annotated-threads``` or ```all-rnr-annotated-threads-with-retweets```.

* ```all-rnr-annotated-threads``` link contains original context dataset in "6392078" veracity dataset ``PHEME_veracity.tar.bz2`` released by PHEME which only contains replies in 'reactions' directory

* ```all-rnr-annotated-threads-with-retweets``` link is post-processed by Sooji that contains additional retweets along with original 'reaction' subdirectory

* ```aug-rnr-annotated-threads-retweets```links contains augmented context dataset

We consider any reaction to source tweet as context and can be stored in 'reaction' directory.

For the development, It is recommended to have a separate directory 'social_context_dataset' to maintain the two 
versions of data set and have two symlinks in this subdirectory. 
Context dataset can be switched via configuring social context loader (``src/data_loader.py``)
