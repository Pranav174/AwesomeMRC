# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Model class. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .configuration_auto import (AlbertConfig, BertConfig, CamembertConfig, CTRLConfig,
                                 DistilBertConfig, GPT2Config, OpenAIGPTConfig, RobertaConfig,
                                 TransfoXLConfig, XLMConfig, XLNetConfig, XLMRobertaConfig)

from .modeling_bert import BertModel, BertForMaskedLM, BertForSequenceClassification, BertForQuestionAnswering, \
    BertForTokenClassification, BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_openai import OpenAIGPTModel, OpenAIGPTLMHeadModel, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_ctrl import CTRLModel, CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_transfo_xl import TransfoXLModel, TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_xlnet import XLNetModel, XLNetLMHeadModel, XLNetForSequenceClassification, XLNetForQuestionAnswering, \
    XLNetForTokenClassification, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_xlm import XLMModel, XLMWithLMHeadModel, XLMForSequenceClassification, XLMForQuestionAnswering, \
    XLM_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_roberta import RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification, \
    RobertaForTokenClassification, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_distilbert import DistilBertModel, DistilBertForQuestionAnswering, DistilBertForMaskedLM, \
    DistilBertForSequenceClassification, DistilBertForTokenClassification, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_camembert import CamembertModel, CamembertForMaskedLM, CamembertForSequenceClassification, \
    CamembertForMultipleChoice, CamembertForTokenClassification, CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_albert import AlbertModel, AlbertForMaskedLM, AlbertForSequenceClassification, \
    AlbertForQuestionAnswering, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_t5 import T5Model, T5WithLMHeadModel, T5_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_xlm_roberta import XLMRobertaModel, XLMRobertaForMaskedLM, XLMRobertaForSequenceClassification, \
    XLMRobertaForMultipleChoice, XLMRobertaForTokenClassification, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

from .modeling_utils import PreTrainedModel, SequenceSummary

from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)


ALL_PRETRAINED_MODEL_ARCHIVE_MAP = dict((key, value)
    for pretrained_map in [
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
        CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        T5_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        ]
    for key, value, in pretrained_map.items())


class AutoModel(object):
    r"""
        :class:`~transformers_mrc.AutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        or the `AutoModel.from_config(config)` class methods.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: T5Model (T5 model)
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `albert`: AlbertModel (ALBERT model)
            - contains `camembert`: CamembertModel (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaModel (XLM-RoBERTa model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL  model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModel.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `openai-gpt` configuration class: OpenAIGPTModel (OpenAI GPT model)
                    - isInstance of `gpt2` configuration class: GPT2Model (OpenAI GPT-2 model)
                    - isInstance of `ctrl` configuration class: CTRLModel (Salesforce CTRL  model)
                    - isInstance of `transfo-xl` configuration class: TransfoXLModel (Transformer-XL model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModel.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, DistilBertConfig):
            return DistilBertModel(config)
        elif isinstance(config, RobertaConfig):
            return RobertaModel(config)
        elif isinstance(config, BertConfig):
            return BertModel(config)
        elif isinstance(config, OpenAIGPTConfig):
            return OpenAIGPTModel(config)
        elif isinstance(config, GPT2Config):
            return GPT2Model(config)
        elif isinstance(config, TransfoXLConfig):
            return TransfoXLModel(config)
        elif isinstance(config, XLNetConfig):
            return XLNetModel(config)
        elif isinstance(config, XLMConfig):
            return XLMModel(config)
        elif isinstance(config, CTRLConfig):
            return CTRLModel(config)
        elif isinstance(config, AlbertConfig):
            return AlbertModel(config)
        elif isinstance(config, CamembertConfig):
            return CamembertModel(config)
        elif isinstance(config, XLMRobertaConfig):
            return XLMRobertaModel(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: T5Model (T5 model)
            - contains `distilbert`: DistilBertModel (DistilBERT model)
            - contains `albert`: AlbertModel (ALBERT model)
            - contains `camembert`: CamembertModel (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaModel (XLM-RoBERTa model)
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)
            - contains `ctrl`: CTRLModel (Salesforce CTRL model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and :func:`~transformers_mrc.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers_mrc.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 't5' in pretrained_model_name_or_path:
            return T5Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'distilbert' in pretrained_model_name_or_path:
            return DistilBertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return AlbertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm-roberta' in pretrained_model_name_or_path:
            return XLMRobertaModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return OpenAIGPTModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return GPT2Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TransfoXLModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'ctrl' in pretrained_model_name_or_path:
            return CTRLModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm-roberta', 'xlm', 'roberta, 'ctrl', 'distilbert', 'camembert', 'albert'".format(pretrained_model_name_or_path))


class AutoModelWithLMHead(object):
    r"""
        :class:`~transformers_mrc.AutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: T5ModelWithLMHead (T5 model)
            - contains `distilbert`: DistilBertForMaskedLM (DistilBERT model)
            - contains `albert`: AlbertForMaskedLM (ALBERT model)
            - contains `camembert`: CamembertForMaskedLM (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaForMaskedLM (XLM-RoBERTa model)
            - contains `roberta`: RobertaForMaskedLM (RoBERTa model)
            - contains `bert`: BertForMaskedLM (Bert model)
            - contains `openai-gpt`: OpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: GPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: XLNetLMHeadModel (XLNet model)
            - contains `xlm`: XLMWithLMHeadModel (XLM model)
            - contains `ctrl`: CTRLLMHeadModel (Salesforce CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModelWithLMHead is designed to be instantiated "
            "using the `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelWithLMHead.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `openai-gpt` configuration class: OpenAIGPTModel (OpenAI GPT model)
                    - isInstance of `gpt2` configuration class: GPT2Model (OpenAI GPT-2 model)
                    - isInstance of `ctrl` configuration class: CTRLModel (Salesforce CTRL  model)
                    - isInstance of `transfo-xl` configuration class: TransfoXLModel (Transformer-XL model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelWithLMHead.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, DistilBertConfig):
            return DistilBertForMaskedLM(config)
        elif isinstance(config, RobertaConfig):
            return RobertaForMaskedLM(config)
        elif isinstance(config, BertConfig):
            return BertForMaskedLM(config)
        elif isinstance(config, OpenAIGPTConfig):
            return OpenAIGPTLMHeadModel(config)
        elif isinstance(config, GPT2Config):
            return GPT2LMHeadModel(config)
        elif isinstance(config, TransfoXLConfig):
            return TransfoXLLMHeadModel(config)
        elif isinstance(config, XLNetConfig):
            return XLNetLMHeadModel(config)
        elif isinstance(config, XLMConfig):
            return XLMWithLMHeadModel(config)
        elif isinstance(config, CTRLConfig):
            return CTRLLMHeadModel(config)
        elif isinstance(config, XLMRobertaConfig):
            return XLMRobertaForMaskedLM(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: T5ModelWithLMHead (T5 model)
            - contains `distilbert`: DistilBertForMaskedLM (DistilBERT model)
            - contains `albert`: AlbertForMaskedLM (ALBERT model)
            - contains `camembert`: CamembertForMaskedLM (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaForMaskedLM (XLM-RoBERTa model)
            - contains `roberta`: RobertaForMaskedLM (RoBERTa model)
            - contains `bert`: BertForMaskedLM (Bert model)
            - contains `openai-gpt`: OpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: GPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: XLNetLMHeadModel (XLNet model)
            - contains `xlm`: XLMWithLMHeadModel (XLM model)
            - contains `ctrl`: CTRLLMHeadModel (Salesforce CTRL model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and :func:`~transformers_mrc.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.
            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers_mrc.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelWithLMHead.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 't5' in pretrained_model_name_or_path:
            return T5WithLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm-roberta' in pretrained_model_name_or_path:
            return XLMRobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return OpenAIGPTLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TransfoXLLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMWithLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'ctrl' in pretrained_model_name_or_path:
            return CTRLLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm-roberta', 'xlm', 'roberta','ctrl', 'distilbert', 'camembert', 'albert'".format(pretrained_model_name_or_path))


class AutoModelForSequenceClassification(object):
    r"""
        :class:`~transformers_mrc.AutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForSequenceClassification (DistilBERT model)
            - contains `albert`: AlbertForSequenceClassification (ALBERT model)
            - contains `camembert`: CamembertForSequenceClassification (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaForSequenceClassification (XLM-RoBERTa model)
            - contains `roberta`: RobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: BertForSequenceClassification (Bert model)
            - contains `xlnet`: XLNetForSequenceClassification (XLNet model)
            - contains `xlm`: XLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModelForSequenceClassification is designed to be instantiated "
            "using the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSequenceClassification.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, AlbertConfig):
            return AlbertForSequenceClassification(config)
        elif isinstance(config, CamembertConfig):
            return CamembertForSequenceClassification(config)
        elif isinstance(config, DistilBertConfig):
            return DistilBertForSequenceClassification(config)
        elif isinstance(config, RobertaConfig):
            return RobertaForSequenceClassification(config)
        elif isinstance(config, BertConfig):
            return BertForSequenceClassification(config)
        elif isinstance(config, XLNetConfig):
            return XLNetForSequenceClassification(config)
        elif isinstance(config, XLMConfig):
            return XLMForSequenceClassification(config)
        elif isinstance(config, XLMRobertaConfig):
            return XLMRobertaForSequenceClassification(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForSequenceClassification (DistilBERT model)
            - contains `albert`: AlbertForSequenceClassification (ALBERT model)
            - contains `camembert`: CamembertForSequenceClassification (CamemBERT model)
            - contains `xlm-roberta`: XLMRobertaForSequenceClassification (XLM-RoBERTa model)
            - contains `roberta`: RobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: BertForSequenceClassification (Bert model)
            - contains `xlnet`: XLNetForSequenceClassification (XLNet model)
            - contains `xlm`: XLMForSequenceClassification (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and :func:`~transformers_mrc.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers_mrc.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return AlbertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'camembert' in pretrained_model_name_or_path:
            return CamembertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm-roberta' in pretrained_model_name_or_path:
            return XLMRobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'xlnet', 'xlm-roberta', 'xlm', 'roberta', 'distilbert', 'camembert', 'albert'".format(pretrained_model_name_or_path))


class AutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformers_mrc.AutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForQuestionAnswering (DistilBERT model)
            - contains `albert`: AlbertForQuestionAnswering (ALBERT model)
            - contains `bert`: BertForQuestionAnswering (Bert model)
            - contains `xlnet`: XLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: XLMForQuestionAnswering (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("AutoModelForQuestionAnswering is designed to be instantiated "
            "using the `AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForQuestionAnswering.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, AlbertConfig):
            return AlbertForQuestionAnswering(config)
        elif isinstance(config, DistilBertConfig):
            return DistilBertForQuestionAnswering(config)
        elif isinstance(config, BertConfig):
            return BertForQuestionAnswering(config)
        elif isinstance(config, XLNetConfig):
            return XLNetForQuestionAnswering(config)
        elif isinstance(config, XLMConfig):
            return XLMForQuestionAnswering(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForQuestionAnswering (DistilBERT model)
            - contains `albert`: AlbertForQuestionAnswering (ALBERT model)
            - contains `bert`: BertForQuestionAnswering (Bert model)
            - contains `xlnet`: XLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: XLMForQuestionAnswering (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and :func:`~transformers_mrc.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers_mrc.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForQuestionAnswering.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return AlbertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return XLMForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'xlnet', 'xlm', 'distilbert', 'albert'".format(pretrained_model_name_or_path))


class AutoModelForTokenClassification:
    def __init__(self):
        raise EnvironmentError("AutoModelForTokenClassification is designed to be instantiated "
                               "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
                               "`AutoModelForTokenClassification.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.
    
            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `camembert` configuration class: CamembertModel (Camembert model)
                    - isInstance of `roberta` configuration class: RobertaModel (Roberta model)

        Examples::
    
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForTokenClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, CamembertConfig):
            return CamembertForTokenClassification(config)
        elif isinstance(config, DistilBertConfig):
            return DistilBertForTokenClassification(config)
        elif isinstance(config, BertConfig):
            return BertForTokenClassification(config)
        elif isinstance(config, XLNetConfig):
            return XLNetForTokenClassification(config)
        elif isinstance(config, RobertaConfig):
            return RobertaForTokenClassification(config)
        elif isinstance(config, XLMRobertaConfig):
            return XLMRobertaForTokenClassification(config)
        raise ValueError("Unrecognized configuration class {}".format(config))
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: DistilBertForTokenClassification (DistilBERT model)
            - contains `camembert`: CamembertForTokenClassification (Camembert model)
            - contains `bert`: BertForTokenClassification (Bert model)
            - contains `xlnet`: XLNetForTokenClassification (XLNet model)
            - contains `roberta`: RobertaForTokenClassification (Roberta model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers_mrc.PreTrainedModel.save_pretrained` and :func:`~transformers_mrc.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers_mrc.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForTokenClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'camembert' in pretrained_model_name_or_path:
            return CamembertForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'distilbert' in pretrained_model_name_or_path:
            return DistilBertForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm-roberta' in pretrained_model_name_or_path:
            return XLMRobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return RobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return BertForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return XLNetForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'xlnet', 'camembert', 'distilbert', 'xlm-roberta', 'roberta'".format(pretrained_model_name_or_path))
