.PHONY = install activate \
	experiments_01 experiments_01-upperbound experiments_01-lowerbound \
	greedy_02 greedy_02-eng greedy_02-por greedy_02-pol greedy_02-rus greedy_02-ara greedy_02-fin


install:
	conda env create -f environment.yml
	conda activate flexible-probing

activate:
	conda activate flexible-probing

## BATCH 1 EXPERIMENTS (aka. validate method using logistic regression probe)
# FUNCTIONS
manual_01_output = experiments/01/01_$(1)-$(2)-$(3)-results.json
manual_01_command = python run.py --language $(1) --attribute "$(2)" --trainer $(3) --gpu --output-file "$(call manual_01_output,$(1),$(2),$(3))" \
					--l1-weight 1e-5 --l2-weight 1e-5 file --file experiments/01_dimension-tests.json

manual_01_outputs_list = $(call manual_01_output,eng,Number,$(1)) $(call manual_01_output,por,Gender\ and\ Noun\ Class,$(1)) \
						 $(call manual_01_output,pol,Tense,$(1)) $(call manual_01_output,rus,Voice,$(1)) \
						 $(call manual_01_output,ara,Case,$(1)) $(call manual_01_output,fin,Case,$(1))

# RULES
LANGUAGES = eng por pol rus ara fin
PROPERTIES = Gender Case Number Animacy Tense Aspect Voice Mood Definiteness Person
TRAINERS = upperbound lowerbound poisson qda conditional-poisson

# Needs special attention because of the spacing in the filename
### experiments rule
# arg1: lang
define experiments_Gender_RULE
experiments/01/01_$(1)-Gender\ and\ Noun\ Class-%-results.json:
	$$(call manual_01_command,$(1),Gender and Noun Class,$$*)

experiments/02/02_$(1)-Gender\ and\ Noun\ Class-%-results.json:
	$$(call greedy_02_command,$(1),Gender and Noun Class,$$*)
endef
$(foreach language,$(LANGUAGES),$(eval $(call experiments_Gender_RULE,$(language))))

### experiments rule
# arg1: lang
# arg2: attribute
# arg3: trainer
define experiments_RULE
experiments/01/01_$(1)-$(2)-$(3)-results.json:
	$$(call manual_01_command,$(1),$(2),$(3))

experiments/02/02_$(1)-$(2)-$(3)-results.json:
	$$(call greedy_02_command,$(1),$(2),$(3))
endef
$(foreach property, $(PROPERTIES), $(foreach language, $(LANGUAGES), $(foreach trainer,$(TRAINERS),$(eval $(call experiments_RULE,$(language),$(property),$(trainer))))))


define experiments_01_output_RULE
manual_01_outputs_trainer_eng_$(1) = $$(call manual_01_output,eng,Number,$(1)) $$(call manual_01_output,eng,Tense,$(1))
manual_01_outputs_trainer_por_$(1) = $$(call manual_01_output,por,Number,$(1)) $$(call manual_01_output,por,Gender\ and\ Noun\ Class,$(1)) \
									 $$(call manual_01_output,por,Tense,$(1))
manual_01_outputs_trainer_pol_$(1) = $$(call manual_01_output,pol,Case,$(1)) $$(call manual_01_output,pol,Number,$(1)) \
									 $$(call manual_01_output,pol,Gender\ and\ Noun\ Class,$(1)) $$(call manual_01_output,pol,Animacy,$(1)) \
								 	 $$(call manual_01_output,pol,Tense,$(1))
manual_01_outputs_trainer_ara_$(1) = $$(call manual_01_output,ara,Number,$(1)) $$(call manual_01_output,ara,Gender\ and\ Noun\ Class,$(1)) \
									 $$(call manual_01_output,ara,Mood,$(1)) $$(call manual_01_output,ara,Aspect,$(1)) $$(call manual_01_output,ara,Voice,$(1)) \
									 $$(call manual_01_output,ara,Case,$(1)) $$(call manual_01_output,ara,Definiteness,$(1))
manual_01_outputs_trainer_fin_$(1) = $$(call manual_01_output,fin,Number,$(1)) $$(call manual_01_output,fin,Case,$(1)) \
									 $$(call manual_01_output,fin,Voice,$(1)) $$(call manual_01_output,fin,Tense,$(1)) \
									 $$(call manual_01_output,fin,Person,$(1))
manual_01_outputs_trainer_rus_$(1) = $$(call manual_01_output,rus,Animacy,$(1)) \
									 $$(call manual_01_output,rus,Gender\ and\ Noun\ Class,$(1)) $$(call manual_01_output,rus,Number,$(1)) \
									 $$(call manual_01_output,rus,Tense,$(1)) $$(call manual_01_output,rus,Aspect,$(1)) \
									 $$(call manual_01_output,rus,Voice,$(1)) $$(call manual_01_output,rus,Case,$(1))
manual_01_outputs_trainer_$(1) = $$(manual_01_outputs_trainer_eng_$(1)) $$(manual_01_outputs_trainer_por_$(1)) $$(manual_01_outputs_trainer_pol_$(1)) \
	$$(manual_01_outputs_trainer_ara_$(1)) $$(manual_01_outputs_trainer_fin_$(1)) $$(manual_01_outputs_trainer_rus_$(1))

experiments_01-$(1): $$(manual_01_outputs_trainer_$(1))
endef
$(foreach trainer,$(TRAINERS),$(eval $(call experiments_01_output_RULE,$(trainer))))

# experiments_01: experiments_01-upperbound experiments_01-lowerbound experiments_01-poisson
# Intractable to run upperbound
experiments_01: experiments_01-lowerbound experiments_01-poisson experiments_01-qda experiments_01-conditional-poisson

## BATCH 2 EXPERIMENTS
greedy_02_output = experiments/02/02_$(1)-$(2)-$(3)-results.json
greedy_02_command = python run.py --language $(1) --attribute "$(2)" --trainer $(3) --gpu --output-file "$(call greedy_02_output,$(1),$(2),$(3))" \
					--l1-weight 1e-5 --l2-weight 1e-5 --wandb --wandb-tag run-new-3 greedy --selection-criterion mi --selection-size 100

experiments_02: experiments_02-eng experiments_02-por experiments_02-pol experiments_02-rus experiments_02-ara experiments_02-fin

# English: [Number, Tense]
greedy_02_outputs_trainer_eng = $(call greedy_02_output,eng,Number,$(1)) $(call greedy_02_output,eng,Tense,$(1))
experiments_02-eng: $(call greedy_02_outputs_trainer_eng,qda) $(call greedy_02_outputs_trainer_eng,poisson) \
	$(call greedy_02_outputs_trainer_eng,lowerbound) $(call greedy_02_outputs_trainer_eng,conditional-poisson)
experiments/02/02_eng-%-qda-results.json:
	$(call greedy_02_command,eng,$*,qda)
experiments/02/02_eng-%-poisson-results.json:
	$(call greedy_02_command,eng,$*,poisson)
experiments/02/02_eng-%-conditional-poisson-results.json:
	$(call greedy_02_command,eng,$*,conditional-poisson)

# Portuguese: [Number, Gender and Noun Class, Tense]
greedy_02_outputs_trainer_por = $(call greedy_02_output,por,Number,$(1)) $(call greedy_02_output,por,Gender\ and\ Noun\ Class,$(1)) \
								$(call greedy_02_output,por,Tense,$(1))
experiments_02-por: $(call greedy_02_outputs_trainer_por,qda) $(call greedy_02_outputs_trainer_por,poisson) \
	$(call greedy_02_outputs_trainer_por,lowerbound) $(call greedy_02_outputs_trainer_por,conditional-poisson)
experiments/02/02_por-%-qda-results.json:
	$(call greedy_02_command,por,$*,qda)
experiments/02/02_por-%-poisson-results.json:
	$(call greedy_02_command,por,$*,poisson)
experiments/02/02_por-%-conditional-poisson-results.json:
	$(call greedy_02_command,por,$*,conditional-poisson)

# Polish: [Case, Number, Gender and Noun Class, Animacy, Tense]
greedy_02_outputs_trainer_pol = $(call greedy_02_output,pol,Case,$(1)) $(call greedy_02_output,pol,Number,$(1)) \
								$(call greedy_02_output,pol,Gender\ and\ Noun\ Class,$(1)) $(call greedy_02_output,pol,Animacy,$(1)) \
								$(call greedy_02_output,pol,Tense,$(1))
experiments_02-pol: $(call greedy_02_outputs_trainer_pol,qda) $(call greedy_02_outputs_trainer_pol,poisson) \
 	$(call greedy_02_outputs_trainer_pol,lowerbound) $(call greedy_02_outputs_trainer_pol,conditional-poisson)
experiments/02/02_pol-%-qda-results.json:
	$(call greedy_02_command,pol,$*,qda)
experiments/02/02_pol-%-poisson-results.json:
	$(call greedy_02_command,pol,$*,poisson)
experiments/02/02_pol-%-conditional-poisson-results.json:
	$(call greedy_02_command,pol,$*,conditional-poisson)

# Russian: [Animacy, Gender and Noun Class, Number, Tense, Aspect, Voice, Case]
greedy_02_outputs_trainer_rus = $(call greedy_02_output,rus,Animacy,$(1)) \
								$(call greedy_02_output,rus,Gender\ and\ Noun\ Class,$(1)) $(call greedy_02_output,rus,Number,$(1)) \
								$(call greedy_02_output,rus,Tense,$(1)) $(call greedy_02_output,rus,Aspect,$(1)) \
								$(call greedy_02_output,rus,Voice,$(1)) $(call greedy_02_output,rus,Case,$(1))
experiments_02-rus: $(call greedy_02_outputs_trainer_rus,qda) $(call greedy_02_outputs_trainer_rus,poisson) \
	$(call greedy_02_outputs_trainer_rus,lowerbound) $(call greedy_02_outputs_trainer_rus,conditional-poisson)
experiments/02/02_rus-%-qda-results.json:
	$(call greedy_02_command,rus,$*,qda)
experiments/02/02_rus-%-poisson-results.json:
	$(call greedy_02_command,rus,$*,poisson)
experiments/02/02_rus-%-conditional-poisson-results.json:
	$(call greedy_02_command,rus,$*,conditional-poisson)

# Arabic: [Number, Gender and Noun Class, Mood, Aspect, Voice, Case, Definiteness]
greedy_02_outputs_trainer_ara = $(call greedy_02_output,ara,Number,$(1)) $(call greedy_02_output,ara,Gender\ and\ Noun\ Class,$(1)) \
								$(call greedy_02_output,ara,Mood,$(1)) $(call greedy_02_output,ara,Aspect,$(1)) $(call greedy_02_output,ara,Voice,$(1)) \
								$(call greedy_02_output,ara,Case,$(1)) $(call greedy_02_output,ara,Definiteness,$(1))
experiments_02-ara: $(call greedy_02_outputs_trainer_ara,qda) $(call greedy_02_outputs_trainer_ara,poisson) \
	$(call greedy_02_outputs_trainer_ara,lowerbound) $(call greedy_02_outputs_trainer_ara,conditional-poisson)
experiments/02/02_ara-%-qda-results.json:
	$(call greedy_02_command,ara,$*,qda)
experiments/02/02_ara-%-poisson-results.json:
	$(call greedy_02_command,ara,$*,poisson)
experiments/02/02_ara-%-conditional-poisson-results.json:
	$(call greedy_02_command,ara,$*,conditional-poisson)

# Finnish: [Number, Case, Voice, Tense, Person]
greedy_02_outputs_trainer_fin = $(call greedy_02_output,fin,Number,$(1)) $(call greedy_02_output,fin,Case,$(1)) \
								$(call greedy_02_output,fin,Voice,$(1)) $(call greedy_02_output,fin,Tense,$(1)) \
								$(call greedy_02_output,fin,Person,$(1))
experiments_02-fin: $(call greedy_02_outputs_trainer_fin,qda) $(call greedy_02_outputs_trainer_fin,poisson) \
	$(call greedy_02_outputs_trainer_fin,lowerbound) $(call greedy_02_outputs_trainer_fin,conditional-poisson)
experiments/02/02_fin-%-qda-results.json:
	$(call greedy_02_command,fin,$*,qda)
experiments/02/02_fin-%-poisson-results.json:
	$(call greedy_02_command,fin,$*,poisson)
experiments/02/02_fin-%-conditional-poisson-results.json:
	$(call greedy_02_command,fin,$*,conditional-poisson)


## BATCH 3 EXPERIMENTS (aka. comparison logistic regression probe vs. deeper probes)
# FUNCTIONS
deep_01_output = experiments/03/03_$(1)-$(2)-relu$(3)-results.json
deep_01_command = python run.py --language $(1) --attribute "$(2)" --trainer conditional-poisson --gpu --probe-num-layers $(3) --activation relu \
--output-file "$(call deep_01_output,$(1),$(2),$(3))" --l1-weight 1e-5 --l2-weight 1e-5 file \
--file experiments/01_dimension-tests.json

deep_01_outputs_list = $(call deep_01_output,eng,Number,$(1)) $(call deep_01_output,por,Gender\ and\ Noun\ Class,$(1)) \
						 $(call deep_01_output,pol,Tense,$(1)) $(call deep_01_output,rus,Voice,$(1)) \
						 $(call deep_01_output,ara,Case,$(1)) $(call deep_01_output,fin,Case,$(1))

# RULES
LANGUAGES = eng por pol rus ara fin
PROPERTIES = Gender Case Number Animacy Tense Aspect Voice Mood Definiteness Person
NUM_LAYERS = 2 3

### experiments rule
# arg1: lang
# arg2: attribute
# arg3: num_layers
define experiments_deep_RULE
experiments/03/03_$(1)-$(2)-relu$(3)-results.json:
	$$(call deep_01_command,$(1),$(2),$(3))

endef
$(foreach property, $(PROPERTIES), $(foreach language, $(LANGUAGES), $(foreach probe-num-layers,$(NUM_LAYERS),$(eval $(call experiments_deep_RULE,$(language),$(property),$(probe-num-layers))))))


define experiments_deep_01_output_RULE
deep_01_outputs_probe-num-layers_eng_$(1) = $$(call deep_01_output,eng,Number,$(1)) $$(call deep_01_output,eng,Tense,$(1))
deep_01_outputs_probe-num-layers_por_$(1) = $$(call deep_01_output,por,Number,$(1)) $$(call deep_01_output,por,Gender\ and\ Noun\ Class,$(1)) \
									 $$(call deep_01_output,por,Tense,$(1))
deep_01_outputs_probe-num-layers_pol_$(1) = $$(call deep_01_output,pol,Case,$(1)) $$(call deep_01_output,pol,Number,$(1)) \
									 $$(call deep_01_output,pol,Gender\ and\ Noun\ Class,$(1)) $$(call deep_01_output,pol,Animacy,$(1)) \
								 	 $$(call deep_01_output,pol,Tense,$(1))
deep_01_outputs_probe-num-layers_ara_$(1) = $$(call deep_01_output,ara,Number,$(1)) $$(call deep_01_output,ara,Gender\ and\ Noun\ Class,$(1)) \
									 $$(call deep_01_output,ara,Mood,$(1)) $$(call deep_01_output,ara,Aspect,$(1)) $$(call deep_01_output,ara,Voice,$(1)) \
									 $$(call deep_01_output,ara,Case,$(1)) $$(call deep_01_output,ara,Definiteness,$(1))
deep_01_outputs_probe-num-layers_fin_$(1) = $$(call deep_01_output,fin,Number,$(1)) $$(call deep_01_output,fin,Case,$(1)) \
									 $$(call deep_01_output,fin,Voice,$(1)) $$(call deep_01_output,fin,Tense,$(1)) \
									 $$(call deep_01_output,fin,Person,$(1))
deep_01_outputs_probe-num-layers_rus_$(1) = $$(call deep_01_output,rus,Animacy,$(1)) \
									 $$(call deep_01_output,rus,Gender\ and\ Noun\ Class,$(1)) $$(call deep_01_output,rus,Number,$(1)) \
									 $$(call deep_01_output,rus,Tense,$(1)) $$(call deep_01_output,rus,Aspect,$(1)) \
									 $$(call deep_01_output,rus,Voice,$(1)) $$(call deep_01_output,rus,Case,$(1))
deep_01_outputs_probe-num-layers_$(1) = $$(deep_01_outputs_probe-num-layers_eng_$(1)) $$(deep_01_outputs_probe-num-layers_por_$(1)) $$(deep_01_outputs_probe-num-layers_pol_$(1)) \
	$$(deep_01_outputs_probe-num-layers_ara_$(1)) $$(deep_01_outputs_probe-num-layers_fin_$(1)) $$(deep_01_outputs_probe-num-layers_rus_$(1))

experiments_deep_01-$(1): $$(deep_01_outputs_probe-num-layers_$(1))
endef
$(foreach probe-num-layers,$(NUM_LAYERS),$(eval $(call experiments_deep_01_output_RULE,$(probe-num-layers))))

# experiments_deep_01:  for relu with 2 and 3 layers
experiments_deep_01: experiments_deep_01-2 experiments_deep_01-3
