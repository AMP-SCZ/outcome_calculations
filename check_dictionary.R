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
dict4 <- read.csv('/data/predict1/home/np487/amp_scz/outcome_calculations/dicts/psychs_outcomes_for_NDA_Nora_updated_20230605.csv')

dict4_added <- dict4 %>%
  filter(!ELEMENT_NAME %in% dict2$ELEMENT_NAME)%>%
  filter(!ELEMENT_NAME %in% c('interview_date', 'ampscz_missing', 'ampscz_missing_spec'))

dict <- dict1 %>%
  rbind(., dict2)%>%
  rbind(., dict3)%>%
  rbind(., dict4_added)%>%
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
  filter(ELEMENT_NAME %in% vars1$variable)%>%
  mutate(ALIASES = case_when(is.na(ALIASES) ~ '',
                             TRUE ~ ALIASES))

dict_additional_outcomes <- dict %>%
  filter(!ELEMENT_NAME %in% vars1$variable)

dict_outcomes_calculated %>%
  filter(!ELEMENT_NAME %in% vars1$variable)%>%
  dplyr::select(ELEMENT_NAME)

dict_additional_outcomes %>%
  filter(!ELEMENT_NAME %in% vars1$variable)%>%
  dplyr::select(ELEMENT_NAME)

vars1 %>%
  filter(!variable %in% dict_outcomes_calculated$ELEMENT_NAME)%>%
  dplyr::select(variable)

# we calculate 13 variables that are not in dict_outcomes_calculated
# we have 0 outcomes in dict_outcomes_calculated that are not included in outcome-calculations
# we have 91 outcomes in additional dictionary that are nto in outcome calculated

write.csv(dict_outcomes_calculated, '/data/predict1/home/np487/amp_scz/outcome_calculations/dicts/dict_calculated_nda3.csv', row.names = FALSE)