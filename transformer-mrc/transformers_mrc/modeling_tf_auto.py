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

from .configuration_auto import (BertConfig, CTRLConfig, DistilBertConfig,
                                 GPT2Config, OpenAIGPTConfig, RobertaConfig,
                                 TransfoXLConfig, XLMConfig, XLNetConfig)

from .modeling_tf_bert import TFBertModel, TFBertForMaskedLM, TFBertForSequenceClassification, \
    TFBertForQuestionAnswering, TFBertForTokenClassification, TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_openai import TFOpenAIGPTModel, TFOpenAIGPTLMHeadModel, TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_gpt2 import TFGPT2Model, TFGPT2LMHeadModel, TF_GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_transfo_xl import TFTransfoXLModel, TFTransfoXLLMHeadModel, TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_xlnet import TFXLNetModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification, \
    TFXLNetForQuestionAnsweringSimple, TFXLNetForTokenClassification, TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_xlm import TFXLMModel, TFXLMWithLMHeadModel, TFXLMForSequenceClassification, \
    TFXLMForQuestionAnsweringSimple, TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_roberta import TFRobertaModel, TFRobertaForMaskedLM, TFRobertaForSequenceClassification, \
    TFRobertaForTokenClassification, TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_distilbert import TFDistilBertModel, TFDistilBertForQuestionAnswering, TFDistilBertForMaskedLM, TFDistilBertForSequenceClassification, TFDistilBertForTokenClassification, TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_ctrl import TFCTRLModel, TFCTRLLMHeadModel, TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_albert import TFAlbertModel, TFAlbertForMaskedLM, TFAlbertForSequenceClassification, TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .modeling_tf_t5 import TFT5Model, TFT5WithLMHeadModel, TF_T5_PRETRAINED_MODEL_ARCHIVE_MAP

from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)


TF_ALL_PRETRAINED_MODEL_ARCHIVE_MAP = dict((key, value)
    for pretrained_map in [
        TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TF_T5_PRETRAINED_MODEL_ARCHIVE_MAP,
        ]
    for key, value, in pretrained_map.items())


class TFAutoModel(object):
    r"""
        :class:`~transformers_mrc.TFAutoModel` is a generic model class
        that will be instantiated as one of the base model classes of the library
        when created with the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The base model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: TFT5Model (T5 model)
            - contains `distilbert`: TFDistilBertModel (DistilBERT model)
            - contains `roberta`: TFRobertaModel (RoBERTa model)
            - contains `bert`: TFBertModel (Bert model)
            - contains `openai-gpt`: TFOpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: TFGPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TFTransfoXLModel (Transformer-XL model)
            - contains `xlnet`: TFXLNetModel (XLNet model)
            - contains `xlm`: TFXLMModel (XLM model)
            - contains `ctrl`: TFCTRLModel (CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("TFAutoModel is designed to be instantiated "
            "using the `TFAutoModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModel.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `distilbert` configuration class: TFDistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: TFRobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: TFBertModel (Bert model)
                    - isInstance of `openai-gpt` configuration class: TFOpenAIGPTModel (OpenAI GPT model)
                    - isInstance of `gpt2` configuration class: TFGPT2Model (OpenAI GPT-2 model)
                    - isInstance of `ctrl` configuration class: TFCTRLModel (Salesforce CTRL  model)
                    - isInstance of `transfo-xl` configuration class: TFTransfoXLModel (Transformer-XL model)
                    - isInstance of `xlnet` configuration class: TFXLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: TFXLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModel.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, DistilBertConfig):
            return TFDistilBertModel(config)
        elif isinstance(config, RobertaConfig):
            return TFRobertaModel(config)
        elif isinstance(config, BertConfig):
            return TFBertModel(config)
        elif isinstance(config, OpenAIGPTConfig):
            return TFOpenAIGPTModel(config)
        elif isinstance(config, GPT2Config):
            return TFGPT2Model(config)
        elif isinstance(config, TransfoXLConfig):
            return TFTransfoXLModel(config)
        elif isinstance(config, XLNetConfig):
            return TFXLNetModel(config)
        elif isinstance(config, XLMConfig):
            return TFXLMModel(config)
        elif isinstance(config, CTRLConfig):
            return TFCTRLModel(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: TFT5Model (T5 model)
            - contains `distilbert`: TFDistilBertModel (DistilBERT model)
            - contains `roberta`: TFRobertaModel (RoBERTa model)
            - contains `bert`: TFTFBertModel (Bert model)
            - contains `openai-gpt`: TFOpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: TFGPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TFTransfoXLModel (Transformer-XL model)
            - contains `xlnet`: TFXLNetModel (XLNet model)
            - contains `ctrl`: TFCTRLModel (CTRL model)

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            model = TFAutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModel.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        if 't5' in pretrained_model_name_or_path:
            return TFT5Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'distilbert' in pretrained_model_name_or_path:
            return TFDistilBertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return TFAlbertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return TFRobertaModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return TFBertModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return TFOpenAIGPTModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return TFGPT2Model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TFTransfoXLModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return TFXLNetModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return TFXLMModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'ctrl' in pretrained_model_name_or_path:
            return TFCTRLModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'distilbert', 'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta', 'ctrl'".format(pretrained_model_name_or_path))


class TFAutoModelWithLMHead(object):
    r"""
        :class:`~transformers_mrc.TFAutoModelWithLMHead` is a generic model class
        that will be instantiated as one of the language modeling model classes of the library
        when created with the `TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: TFT5WithLMHeadModel (T5 model)
            - contains `distilbert`: TFDistilBertForMaskedLM (DistilBERT model)
            - contains `roberta`: TFRobertaForMaskedLM (RoBERTa model)
            - contains `bert`: TFBertForMaskedLM (Bert model)
            - contains `openai-gpt`: TFOpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: TFGPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `transfo-xl`: TFTransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: TFXLNetLMHeadModel (XLNet model)
            - contains `xlm`: TFXLMWithLMHeadModel (XLM model)
            - contains `ctrl`: TFCTRLLMHeadModel (CTRL model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("TFAutoModelWithLMHead is designed to be instantiated "
            "using the `TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelWithLMHead.from_config(config)` methods.")

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
            return TFDistilBertForMaskedLM(config)
        elif isinstance(config, RobertaConfig):
            return TFRobertaForMaskedLM(config)
        elif isinstance(config, BertConfig):
            return TFBertForMaskedLM(config)
        elif isinstance(config, OpenAIGPTConfig):
            return TFOpenAIGPTLMHeadModel(config)
        elif isinstance(config, GPT2Config):
            return TFGPT2LMHeadModel(config)
        elif isinstance(config, TransfoXLConfig):
            return TFTransfoXLLMHeadModel(config)
        elif isinstance(config, XLNetConfig):
            return TFXLNetLMHeadModel(config)
        elif isinstance(config, XLMConfig):
            return TFXLMWithLMHeadModel(config)
        elif isinstance(config, CTRLConfig):
            return TFCTRLLMHeadModel(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the language modeling model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `t5`: TFT5WithLMHeadModel (T5 model)
            - contains `distilbert`: TFDistilBertForMaskedLM (DistilBERT model)
            - contains `roberta`: TFRobertaForMaskedLM (RoBERTa model)
            - contains `bert`: TFBertForMaskedLM (Bert model)
            - contains `openai-gpt`: TFOpenAIGPTLMHeadModel (OpenAI GPT model)
            - contains `gpt2`: TFGPT2LMHeadModel (OpenAI GPT-2 model)
            - contains `transfo-xl`: TFTransfoXLLMHeadModel (Transformer-XL model)
            - contains `xlnet`: TFXLNetLMHeadModel (XLNet model)
            - contains `xlm`: TFXLMWithLMHeadModel (XLM model)
            - contains `ctrl`: TFCTRLLMHeadModel (CTRL model)

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelWithLMHead.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        if 't5' in pretrained_model_name_or_path:
            return TFT5WithLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'distilbert' in pretrained_model_name_or_path:
            return TFDistilBertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return TFAlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return TFRobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return TFBertForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'openai-gpt' in pretrained_model_name_or_path:
            return TFOpenAIGPTLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'gpt2' in pretrained_model_name_or_path:
            return TFGPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'transfo-xl' in pretrained_model_name_or_path:
            return TFTransfoXLLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return TFXLNetLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return TFXLMWithLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'ctrl' in pretrained_model_name_or_path:
            return TFCTRLLMHeadModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'distilbert', 'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', "
                         "'xlm', 'roberta', 'ctrl'".format(pretrained_model_name_or_path))


class TFAutoModelForSequenceClassification(object):
    r"""
        :class:`~transformers_mrc.TFAutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - contains `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: TFBertForSequenceClassification (Bert model)
            - contains `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - contains `xlm`: TFXLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("TFAutoModelForSequenceClassification is designed to be instantiated "
            "using the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForSequenceClassification.from_config(config)` methods.")

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
        if isinstance(config, DistilBertConfig):
            return TFDistilBertForSequenceClassification(config)
        elif isinstance(config, RobertaConfig):
            return TFRobertaForSequenceClassification(config)
        elif isinstance(config, BertConfig):
            return TFBertForSequenceClassification(config)
        elif isinstance(config, XLNetConfig):
            return TFXLNetForSequenceClassification(config)
        elif isinstance(config, XLMConfig):
            return TFXLMForSequenceClassification(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - contains `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: TFBertForSequenceClassification (Bert model)
            - contains `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - contains `xlm`: TFXLMForSequenceClassification (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForSequenceClassification.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return TFDistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'albert' in pretrained_model_name_or_path:
            return TFAlbertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return TFRobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return TFBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return TFXLNetForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return TFXLMForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'distilbert', 'bert', 'xlnet', 'xlm', 'roberta'".format(pretrained_model_name_or_path))


class TFAutoModelForQuestionAnswering(object):
    r"""
        :class:`~transformers_mrc.TFAutoModelForQuestionAnswering` is a generic model class
        that will be instantiated as one of the question answering model classes of the library
        when created with the `TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForQuestionAnswering (DistilBERT model)
            - contains `bert`: TFBertForQuestionAnswering (Bert model)
            - contains `xlnet`: TFXLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: TFXLMForQuestionAnswering (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError("TFAutoModelForQuestionAnswering is designed to be instantiated "
            "using the `TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForQuestionAnswering.from_config(config)` methods.")

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
        if isinstance(config, DistilBertConfig):
            return TFDistilBertForQuestionAnswering(config)
        elif isinstance(config, BertConfig):
            return TFBertForQuestionAnswering(config)
        elif isinstance(config, XLNetConfig):
            return TFXLNetForQuestionAnswering(config)
        elif isinstance(config, XLMConfig):
            return TFXLMForQuestionAnswering(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: TFDistilBertForQuestionAnswering (DistilBERT model)
            - contains `bert`: TFBertForQuestionAnswering (Bert model)
            - contains `xlnet`: TFXLNetForQuestionAnswering (XLNet model)
            - contains `xlm`: TFXLMForQuestionAnswering (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers_mrc.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

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

            model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForQuestionAnswering.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        if 'distilbert' in pretrained_model_name_or_path:
            return TFDistilBertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'bert' in pretrained_model_name_or_path:
            return TFBertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return TFXLNetForQuestionAnsweringSimple.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlm' in pretrained_model_name_or_path:
            return TFXLMForQuestionAnsweringSimple.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'distilbert', 'bert', 'xlnet', 'xlm'".format(pretrained_model_name_or_path))


class TFAutoModelForTokenClassification:
    def __init__(self):
        raise EnvironmentError("TFAutoModelForTokenClassification is designed to be instantiated "
                               "using the `TFAutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
                               "`AutoModelForTokenClassification.from_config(config)` methods.")

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

            config: (`optional`) instance of a class derived from :class:`~transformers_mrc.PretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBert model)
                    - isInstance of `roberta` configuration class: RobteraModel (Roberta model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForTokenClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if isinstance(config, BertConfig):
            return TFBertForTokenClassification(config)
        elif isinstance(config, XLNetConfig):
            return TFXLNetForTokenClassification(config)
        elif isinstance(config, DistilBertConfig):
            return TFDistilBertForTokenClassification(config)
        elif isinstance(config, RobertaConfig):
            return TFRobertaForTokenClassification(config)
        raise ValueError("Unrecognized configuration class {}".format(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the question answering model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `bert`: BertForTokenClassification (Bert model)
            - contains `xlnet`: XLNetForTokenClassification (XLNet model)
            - contains `distilbert`: DistilBertForTokenClassification (DistilBert model)
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

            model = TFAutoModelForTokenClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForTokenClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForTokenClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForTokenClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        if 'bert' in pretrained_model_name_or_path:
            return TFBertForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'xlnet' in pretrained_model_name_or_path:
            return TFXLNetForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'distilbert' in pretrained_model_name_or_path:
            return TFDistilBertForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        elif 'roberta' in pretrained_model_name_or_path:
            return TFRobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        raise ValueError("Unrecognized model identifier in {}. Should contains one of "
                         "'bert', 'xlnet', 'distilbert', 'roberta'".format(pretrained_model_name_or_path))
