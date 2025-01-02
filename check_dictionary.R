# For NDA3 upload, we want to have all dictionaries within one csv file.
# Tashrif has collected the different dictionaries that we had so far 
# (3 csv-files). We will load all the outcomes here to check that none is missed
# or too much.

library(tidyverse)
dict1 <- read.csv('/data/predict1/home/np487/amp_scz/outcome_calculations/dicts/other_outcomes.csv')%>%
  dplyr::select(-c(Dictionary, CsvFile))%>%
  rename_with(toupper)%>%
  dplyr::rename(ELEMENT_NAME = ELEMENTNAME,
                DATA_TYPE = DATATYPE,
                DESCRIPTION = ELEMENTDESCRIPTION,
                VALUE_RANGE = VALUERANGE)
dict2 <- read.csv('/data/predict1/home/np487/amp_scz/outcome_calculations/dicts/psychs.csv')
dict3 <- read.csv('/data/predict1/home/np487/amp_scz/outcome_calculations/dicts/SCID5_dictionary.csv')

dict <- dict1 %>%
  rbind(., dict2)%>%
  rbind(., dict3)%>%
  # filter out all rows that are completely empty
  filter(if_any(everything(), ~ !is.na(.) & . != ""))

all_outcomes <- read.csv('/data/predict1/home/np487/amp_scz/test_subjects/prescient_new_subjects.csv')

vars1 <- all_outcomes %>%
  distinct(variable, .keep_all = TRUE)

dict %>%
  filter(!ELEMENT_NAME %in% vars1$variable)%>%
  dplyr::select(ELEMENT_NAME)

vars1 %>%
  filter(!variable %in% dict$ELEMENT_NAME)%>%
  dplyr::select(variable)

# we have names (12-24) that are pulled directly from Redcap, therefore 
# it is not needed to have them in the dictionary.

# For the new variables in the PSYCHS we have created the dictionary but we did 
# not generate this data. Instead, PSYCHS data is pulled directly.

dict_outcomes_calculated <- dict %>%
  filter(ELEMENT_NAME %in% vars1$variable)

dict_additional_outcomes <- dict %>%
  filter(!ELEMENT_NAME %in% vars1$variable)

add_outcomes_calc <- data.frame(ELEMENT_NAME = c('sips_bips_scr_lifetime', 
                                                 'sips_aps_scr_lifetime', 
                                                 'sips_grd_scr_lifetime',
                                                 'sips_bips_fu_new',
                                                 'sips_bips_lifetime',
                                                 'sips_aps_fu_new',
                                                 'sips_aps_lifetime',
                                                 'sips_grd_fu_new',
                                                 'sips_chr_lifetime'),
                                DATA_TYPE = c(),
                                DESCRIPTION = c(),
                                VALUE_RANGE = c(),
                                NOTES = c(),
                                ALIASES = c())