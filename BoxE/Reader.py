import tensorflow as tf
import pprint
NewCheck =tf.train.NewCheckpointReader("weights_NELLRuleInjSplit90Mat/values.ckpt")
print("debug_string:\n")
pprint.pprint(NewCheck.debug_string().decode("utf-8"))
print("get_tensor:\n")
pprint.pprint(NewCheck.get_tensor("Variable"))
print("get_variable_to_dtype_map\n")
pprint.pprint(NewCheck.get_variable_to_dtype_map())
print("get_variable_to_shape_map\n")
pprint.pprint(NewCheck.get_variable_to_shape_map())