from predict.distribution import Process
from tokenization import tokenization

class Preprocessor(Process):

    def __init__(self,
                 config,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTPreprocessor',
                 **kwargs):

        kwargs.clear()

        if config.mode.startswith("predict"):
            Process.__init__(
                self, job_name, thread_num, input_queue, output_queue, batch_size=config.predict_batch_size)

        elif config.mode == "preprocess":
            Process.__init__(
                self, job_name, thread_num, input_queue, output_queue, batch_size=config.preprocess_batch_size)

        self.append_tensor_names = []
        if hasattr(config, "append_feature_columns") and config.append_feature_columns is not None:
            for schema in config.append_feature_columns.split(","):
                name = schema.split(":")[0]
                self.append_tensor_names.append(name)

        self.mode = config.mode

    @classmethod
    def get_preprocessor(cls, **kwargs):
        
        json_file = FLAGS.config
        with tf.gfile.GFile(json_file, mode='r') as reader:
            text = reader.read()

        config_dict = json.loads(text)
        for values in config_dict.values():
            if isinstance(values, str):
                continue
            for k, v in values.items():
                    kwargs[k] = v
            kwargs["mode"] = FLAGS.mode

        preprocessor = cls(config_dict, **kwargs)
        return preprocessor

    def set_feature_schema(self):
        raise NotImplementedError("must be implemented in descendants")

    def convert_example_to_features(self, items):
        raise NotImplementedError("must be implemented in descendants")

    def _convert(self, convert_example_to_features, *args):

        # mode check
        if not ("on_the_fly" in self.mode or self.mode == "preprocess"):
            raise ValueError("Please using on_the_fly or preprocess mode")

        batch_features = []
        batch_size = len(args[0])
        for i in range(batch_size):
            items = []
            for feat in args:
                if isinstance(feat[i], np.ndarray):
                    assert feat[i][0] is not None, "In on the fly mode where object is ndarray, column has null value"
                    items.append(feat[i][0])
                else:
                    assert feat[i] is not None, "In on the fly mode, column has null value"
                    items.append(feat[i])
            features = convert_example_to_features(items)
            batch_features.append(features)

        stacked_features = np.stack(batch_features, axis=1)
        concat_features = []
        for i in range(stacked_features.shape[0]):
            concat_features.append(np.asarray(" ".join(stacked_features[i])))
        return concat_features

    # Inputs from Reader's map_batch_prefetch method
    def call(self, inputs):
        self.set_feature_schema()

        items = []
        for name in self.input_tensor_names:
            items.append(inputs[name])

        if not ("on_the_fly" in self.mode or self.mode == "preprocess"):
            return items

        self.Tout = [tf.string] * len(self.seq_lens)

        batch_features = tf.py_func(functools.partial(self._convert,
                                                      self.convert_example_to_features),
                                    items, self.Tout)

        ret = []
        for idx, feature in enumerate(batch_features):
            seq_len = self.seq_lens[idx]
            feature_type = self.feature_value_types[idx]
            if feature_type == tf.int64:
                input_tensor = tf.string_to_number(
                    tf.string_split(tf.expand_dims(feature, axis=0), delimiter=" ").values,
                    tf.int64)
            elif feature_type == tf.float32:
                input_tensor = tf.string_to_number(
                    tf.string_split(tf.expand_dims(feature, axis=0), delimiter=" ").values,
                    tf.float32)
            elif feature_type == tf.string:
                input_tensor = feature
            else:
                raise NotImplementedError

            input_tensor = tf.reshape(input_tensor, [-1, seq_len])
            ret.append(input_tensor)

        for name in self.append_tensor_names:
            ret.append(inputs[name])

        return ret

    def process(self, inputs):
        self.set_feature_schema()

        if isinstance(inputs, dict):
            inputs = [inputs]

        batch_features = []
        for input in inputs:
            items = []
            for name in self.input_tensor_names:
                items.append(input[name])
            features = self.convert_example_to_features(items)
            batch_features.append(features)

        stacked_features = np.stack(batch_features, axis=1)
        concat_features = []
        for i in range(stacked_features.shape[0]):
            concat_features.append(np.asarray(" ".join(stacked_features[i])))

        if self.mode.startswith("predict") or self.mode == "preprocess":
            for name in self.output_schema.split(","):
                if name in self.input_tensor_names:
                    self.output_tensor_names.append(name)

        ret = {}
        for idx, name in enumerate(self.output_tensor_names):
            if idx < len(concat_features):
                feature = concat_features[idx]
                seq_len = self.seq_lens[idx]
                feature_type = self.feature_value_types[idx]
                feature = feature.tolist()
                if feature_type == tf.int64:
                    input_tensor = [int(x) for x in feature.split(" ")]
                elif feature_type == tf.float32:
                    input_tensor = [float(x) for x in feature.split(" ")]
                elif feature_type == tf.string:
                    input_tensor = feature
                else:
                    raise NotImplementedError
                input_tensor = np.reshape(input_tensor, [-1, seq_len])
                name = self.output_tensor_names[idx]
                ret[name] = input_tensor
            else:
                left = []
                for ele in inputs:
                    left.append(ele[name])
                left_tensor = np.asarray(left)
                ret[name] = np.reshape(left_tensor, [-1, 1])

        return ret