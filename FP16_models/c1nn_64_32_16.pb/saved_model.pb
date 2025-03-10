��#
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8�� 
v
fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namefc_1/kernel
o
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel*"
_output_shapes
:@*
dtype0
j
	fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_1/bias
c
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes
:@*
dtype0
v
fc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namefc_2/kernel
o
fc_2/kernel/Read/ReadVariableOpReadVariableOpfc_2/kernel*"
_output_shapes
:@@*
dtype0
j
	fc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_2/bias
c
fc_2/bias/Read/ReadVariableOpReadVariableOp	fc_2/bias*
_output_shapes
:@*
dtype0
v
fc_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_namefc_5/kernel
o
fc_5/kernel/Read/ReadVariableOpReadVariableOpfc_5/kernel*"
_output_shapes
:@ *
dtype0
j
	fc_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	fc_5/bias
c
fc_5/bias/Read/ReadVariableOpReadVariableOp	fc_5/bias*
_output_shapes
: *
dtype0
v
fc_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namefc_6/kernel
o
fc_6/kernel/Read/ReadVariableOpReadVariableOpfc_6/kernel*"
_output_shapes
:  *
dtype0
j
	fc_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	fc_6/bias
c
fc_6/bias/Read/ReadVariableOpReadVariableOp	fc_6/bias*
_output_shapes
: *
dtype0
v
fc_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namefc_9/kernel
o
fc_9/kernel/Read/ReadVariableOpReadVariableOpfc_9/kernel*"
_output_shapes
: *
dtype0
j
	fc_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	fc_9/bias
c
fc_9/bias/Read/ReadVariableOpReadVariableOp	fc_9/bias*
_output_shapes
:*
dtype0
x
fc_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefc_10/kernel
q
 fc_10/kernel/Read/ReadVariableOpReadVariableOpfc_10/kernel*"
_output_shapes
:*
dtype0
l

fc_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc_10/bias
e
fc_10/bias/Read/ReadVariableOpReadVariableOp
fc_10/bias*
_output_shapes
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�
*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
�A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�@
value�@B�@ B�@
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
regularization_losses
trainable_variables
	variables
	keras_api

signatures
#_self_saveable_object_factories
trt_engine_resources
%
#_self_saveable_object_factories
w
regularization_losses
trainable_variables
	variables
	keras_api
#_self_saveable_object_factories
4
	keras_api
# _self_saveable_object_factories
�

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
#'_self_saveable_object_factories
�

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
#._self_saveable_object_factories
w
/regularization_losses
0trainable_variables
1	variables
2	keras_api
#3_self_saveable_object_factories
w
4regularization_losses
5trainable_variables
6	variables
7	keras_api
#8_self_saveable_object_factories
�

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
#?_self_saveable_object_factories
�

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
#F_self_saveable_object_factories
w
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
#K_self_saveable_object_factories
w
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
#P_self_saveable_object_factories
�

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
#W_self_saveable_object_factories
�

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
#^_self_saveable_object_factories
w
_regularization_losses
`trainable_variables
a	variables
b	keras_api
#c_self_saveable_object_factories
w
dregularization_losses
etrainable_variables
f	variables
g	keras_api
#h_self_saveable_object_factories
w
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
#m_self_saveable_object_factories
�

nkernel
obias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
#t_self_saveable_object_factories
 
f
!0
"1
(2
)3
94
:5
@6
A7
Q8
R9
X10
Y11
n12
o13
f
!0
"1
(2
)3
94
:5
@6
A7
Q8
R9
X10
Y11
n12
o13
�
unon_trainable_variables
regularization_losses
vlayer_metrics
wmetrics
xlayer_regularization_losses
trainable_variables

ylayers
	variables
#z_self_saveable_object_factories
 
 
 
 
 
 
 
�
{non_trainable_variables
regularization_losses
|layer_metrics
}metrics
~layer_regularization_losses
trainable_variables

layers
	variables
$�_self_saveable_object_factories
 
&
$�_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
�
�non_trainable_variables
#regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
$trainable_variables
�layers
%	variables
$�_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
�non_trainable_variables
*regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
+trainable_variables
�layers
,	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
/regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
0trainable_variables
�layers
1	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
4regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
5trainable_variables
�layers
6	variables
$�_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
�
�non_trainable_variables
;regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
<trainable_variables
�layers
=	variables
$�_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
�
�non_trainable_variables
Bregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Ctrainable_variables
�layers
D	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
Gregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Htrainable_variables
�layers
I	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
Lregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Mtrainable_variables
�layers
N	variables
$�_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
�
�non_trainable_variables
Sregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Ttrainable_variables
�layers
U	variables
$�_self_saveable_object_factories
 
XV
VARIABLE_VALUEfc_10/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
fc_10/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
�
�non_trainable_variables
Zregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
[trainable_variables
�layers
\	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
_regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
`trainable_variables
�layers
a	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
dregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
etrainable_variables
�layers
f	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
iregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
jtrainable_variables
�layers
k	variables
$�_self_saveable_object_factories
 
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

n0
o1

n0
o1
�
�non_trainable_variables
pregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
qtrainable_variables
�layers
r	variables
$�_self_saveable_object_factories
 
 
 
 
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_inputPlaceholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
PartitionedCallPartitionedCallserving_default_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_2777539
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenamefc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOpfc_2/kernel/Read/ReadVariableOpfc_2/bias/Read/ReadVariableOpfc_5/kernel/Read/ReadVariableOpfc_5/bias/Read/ReadVariableOpfc_6/kernel/Read/ReadVariableOpfc_6/bias/Read/ReadVariableOpfc_9/kernel/Read/ReadVariableOpfc_9/bias/Read/ReadVariableOp fc_10/kernel/Read/ReadVariableOpfc_10/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_2777604
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc_1/kernel	fc_1/biasfc_2/kernel	fc_2/biasfc_5/kernel	fc_5/biasfc_6/kernel	fc_6/biasfc_9/kernel	fc_9/biasfc_10/kernel
fc_10/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_2777656��
��
5
__inference_pruned_2777532	
input
identity�
%Func/StatefulPartitionedCall/input/_0Identityinput*
T0*/
_output_shapes
:���������2'
%Func/StatefulPartitionedCall/input/_0�
+StatefulPartitionedCall/mnist/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2-
+StatefulPartitionedCall/mnist/flatten/Const�
-StatefulPartitionedCall/mnist/flatten/ReshapeReshape.Func/StatefulPartitionedCall/input/_0:output:04StatefulPartitionedCall/mnist/flatten/Const:output:0*
T0*(
_output_shapes
:����������2/
-StatefulPartitionedCall/mnist/flatten/Reshape�
=StatefulPartitionedCall/mnist/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2?
=StatefulPartitionedCall/mnist/tf.expand_dims_3/ExpandDims/dim�
9StatefulPartitionedCall/mnist/tf.expand_dims_3/ExpandDims
ExpandDims6StatefulPartitionedCall/mnist/flatten/Reshape:output:0FStatefulPartitionedCall/mnist/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2;
9StatefulPartitionedCall/mnist/tf.expand_dims_3/ExpandDims�
8StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims
ExpandDimsBStatefulPartitionedCall/mnist/tf.expand_dims_3/ExpandDims:output:0AStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������26
4StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims�
OStatefulPartitionedCall/mnist/fc_1/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2Q
OStatefulPartitionedCall/mnist/fc_1/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer�
OStatefulPartitionedCall/mnist/fc_1/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose=StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims:output:0XStatefulPartitionedCall/mnist/fc_1/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������2Q
OStatefulPartitionedCall/mnist/fc_1/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�
6StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1Const*&
_output_shapes
:@*
dtype0*�
value�B�@*��|>�	�<�^�<���(�����\=�j������=?.���';�ɠ=�
>I�*�)޽���=-T�=���=�OC�����nP�=�!q=�R���Ƭ=����<-ؙ=5e���Z����=i\��oW�@6v��N�=h�>��>r��8�=U��<��<
 ����>���<,)#=��=ܐ.�� �X(�������>�9�=�d�=�u�]*����A=��ټ�դ=�|�tr�=�w>�E%<��<j���f<�뼥/�=l>L��=GC���=�P�E"����=+�\=]3	>�>���=՞��4��=�@>���=.d�=YS�� ���D�xRG=������=(����6�=�
�=��>�hv�(�=`0>�/���������=���<�	�=� ���>�o�=�|��˃���=*�>�)#=~�,>�� �����+��}CN���:�]�=���=�E���n�=��=q/>q�{=��!�tP�=͎>v��=�3>�讽>�g=>t�����:b(��i�=v����͞<�s����	bZ=�q>wn$>~���LE
�m�e��>��=5�A=�o�=�9c��5>ZV>�nr�\�˽k�4<|��m�=���r
��{��n��Cy½B[>���Yƫ=!졻����>�UY=j��=���=�9H��*>76_�OL�=�h7�h�>D�F�1L3>vL����,��=�D�=�6�X=�=� =O�;�����,�=�z�c�=!=��)����=28
6StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_1/conv1dConv2DSStatefulPartitionedCall/mnist/fc_1/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������@�*
data_formatNCHW*
paddingVALID*
strides
2+
)StatefulPartitionedCall/mnist/fc_1/conv1d�
QStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2S
QStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
QStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose2StatefulPartitionedCall/mnist/fc_1/conv1d:output:0ZStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������@2S
QStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_1/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_1/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOpConst*
_output_shapes
:@*
dtype0*�
value�B�@*�    M�{<2�n��u����0=�<=    8��;���:���;���9��0:    �Ƃ:*�:f��;��;���<�ûh��:b<    �lV<    �<<}C;��    E �;-�������	=�x�9�����;h�<�6���;�~&�    �W:�ؖ��'�<L4:����    #��:���.�"=�W5;v[��=;J�: �><#�+��Y:�&=�;;��:f�<���:    ��:2;
9StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_1/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_1/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������@2,
*StatefulPartitionedCall/mnist/fc_1/BiasAdd�
'StatefulPartitionedCall/mnist/fc_1/ReluRelu3StatefulPartitionedCall/mnist/fc_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2)
'StatefulPartitionedCall/mnist/fc_1/Relu�
8StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims
ExpandDims5StatefulPartitionedCall/mnist/fc_1/Relu:activations:0AStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@26
4StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims�
OStatefulPartitionedCall/mnist/fc_2/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2Q
OStatefulPartitionedCall/mnist/fc_2/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer�
OStatefulPartitionedCall/mnist/fc_2/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose=StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims:output:0XStatefulPartitionedCall/mnist/fc_2/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:���������@�2Q
OStatefulPartitionedCall/mnist/fc_2/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizerف
6StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1Const*&
_output_shapes
:@@*
dtype0*��
value��B��@@*��C���wX.�o�j����4�=�A���,�|g�;�+d<}D9�H��<G����z`�h	c=Pռ�`&���6�ol�;a�'�H��<J}9;�T';=�5����=su><Mn�� ��<�7=���?��<	��=�~8;9��=��=�!.<PP�=�I�=ϧ�;3��<
B�<{;�=��<=��=�=;a�=$)�<���j�K=�A=�>>��<����5�=�3�Xct;Zu�%������@��8�����<�q=��N�='%�=�"=N��
�ʽ�F��X��9���=�e�����<B=�J<���M���F�|(�<J�N=bl=��=d��=�Z=�C-��dV<g4�m���·=�b�=89��m+F=�dP<$�=J�=9g��
��w��Ow�<�A�<NI����=Օ<�1�=�>6�=eIB=�<`��M�=�e<=���=�&ս÷;��=��<�@���� =%�$=�7$=.J�"Z<����`�����=������=D�(<��9=���<5
=��4�꩗=��G����=d�S=��>P<�=���Ji�<)Ͼ�U��ޓ>��_E��MȺ<ᯜ�bul<,��=��<.�.�4f�=cV&�r�>Y�л�G>6��<���-��<.��=0@�>�(P;`Q�7�<�B�'�=ٹ�\I>_�t����<���<�a��������E�u��;����U=�ѽ��=M9��>Ǖ=�~���4���=Ṻ=k"���a��n�;���<@����H$=�c��4Y�S ��sr<FU�;Le5=Wt��\�=��Y=��o=X�;P';T�4;A�$4=U����8�<��<�/h���x�.�����"=y�;M�S=x`ͻCǥ=B�=��=��5=�-�;�/=5����̾�\s=D��<��<)��<P=0:L�<=��=�g,={=μYjo��ض���Z�o a�?�ٺ�R��M� ͑��9s�<FB��8�<�{<��f=#�Q��=z�V=lI���%�< ���|=b�=$w���5̽7��=$/<�7U=�.���*=NI�򞲼�f���r޽3�=D���
=>ŕ�h/�=���=�q�=��&�xC��6)��<}>��;�7��Ye��}i�GT�w��g,��Ֆ�Y�=�ч����*>�{	�f�н>wּ���d���+Ƚ��ѽ�8Y=O���R��<p`�=����UG���=�䢕;h��;��9=g������=��W�~7�rB�=�=�A�=�MϽ�c�=��n�I g�Huf=p=���{<c��<p��?t=-4=eh�<Yf�+V�<���;Y�=�]i=ڜ����=a��Z0ɽp<D�=@�==~�=6�=g@&=?�%{=�߶=�X=Kf<�V�=^3�=@B4=�4;r�m=�4c<��Ͻx��-j=9h=��;K� �:ܣ=�߁=iZ>a7�y�7�{�����ƴ�=VF:=�J�=jrѻ��=�*=��<&ON<Br=�bU=��<6���o��J}2=^x���(=��=ZD>��=W7=���I ~��X��z�<	�ν�t��?P�����=5��HսF|O=���Wpz=.�л�м��&>Vb���N@��%J&�E>���;;a��v<��ǽ�%��������䚼W�?=�_����=Di	>��<��g�1�<�$:��s�H�p6
�?�=&��=�9=h{>Nm�#�S�����#'<4\��3B���=��h=G���경��>��>Z=���<��8=��ͽ������W=N����<[Ԧ���<�-�id�U�F=
i�{�;�t[<�'�)��T+=�2�<���<ǉ���躽_޻��|��ё�3eF=�����w.�O��=���<��ǻp��-1�<B���1=������%<�%���<�ւ�����o�=�uܻ# =z�=|�=���������Z����2�E�G5G=@�<q<�7�=��5���{=S;Ew<��9>����
7=��	�=*��C�=RL;�^�<w[S=�-�;J�U;^�=�0�"<(=�L � ͌<<���=ҩ8=JFO�G"�<Q{�<o�<X��-;�����=��)��ת<�=�=��<�5=�+��$$�<C'd=`Һ<X�=�1>�M=<�>G�{�
�
;�)@=�ҽ��=��y=r��<�e�<��s	1=�$L<��>x7��\�;�,<��j��b}�=���<P1>�q�=�3=z8q=�pH�e��=�j�=|2*�Q��=��N�W��;$GV<jV*�HY=w�(=|�'��F�=�z�=��=��<}�=�ꮽ1|�=;Ǹ�ui}=�t?��=���<�h=ӹ�=&���j�q=|4;ר�=T��<�̋��p�=x�=?pV��p=�T=
_�=.�N=Y��g�O=�;=!(�=bs=��[<5f��z�m�[g =�'V=��Ž�N[���=�6=egc<쉇=�Do��[M=$��Z�R�<|Ͽ=�~,>=�F|�����C�����=!V���
����<@��#]&=�:,��}0��ߔ=�(N�+��<�&=0g=B.�<��<[�-=`=<�<�-=�w==��
<Jp{;�Q=3� �I��`�=ˈս�Un=��=1f?=��μ����=�=!�[=A)���h���r0=ޖ=OVR<A��<� �s�"= 9[�YeQ=e�:��>&�ʺ�ݪ=c��=ĺ�;����}z��h�=%倽��=�o�=qɫ=f�">\�~=�x>����<P���i=��$=xӻ���'�<���=�U���=�P�=�خ�JP�<4+-��M�=� c<�!�=O�<�*�=È���6,>2nF>=�=K\�=;���>�����_���>z'��>=*h">ra�;��[=��!>��=U�> �>$��=�>�M>�>��7��C�H�K<� ��^�>06)>l��=Kν��_=��]>�h��{$>7�$��8>�Q�<qLU�&Z$>Q�
>u�9>'�>��F=�^.��x<�0��L%>$I�=���g-���b_=T
>�+&>�K=��=�V���=�
%<M��<	[�;sN<qWn��2��;ݽI��E񹼚'>��/>��:=*P<��*�iNy<n4��%���(������<����8=�6�>�OԼ䴟=����H
� �h�3$z=�ѓ=�w*=^8���;eD��O>'8M=��=��=/�j�7p�+P&;%��>��y��v��gnP����=�V >S�=#��"5���v����h�S�A���-=�A>5��<�νI���~z�z<�=~��>�8�b�H�+��U9��My����=��8��I���f���� =3��<�`Z<�柽��ͺ���J�^�V����$<�k�;��j<Z2����6��<���@��<z!���9�����"Z<3�=�!�<��ں��<ᕸ�W��:��%�� �=pK�<�Z<_�^�=�R ��ʽ���������<2�<��Q<;��<zi�=�Ҫ�(
����"=N���zT=��;�L�gԼY�8�
��=��N:
��<� ��T�'*�<W'��}=����Up�����|��=��u=��=�$��Bx���<�=��<U������~�<h�U=a��=��&<0=Z�<�a��l���	$=�y='��<VO�굚<�J�=9@�<��>�f<qa!��<�����u��B=v�#<��}=DEP=�WG:��>r)u=����齺L�.�b��=��ļ�>=y�(=�b�=v�S=c�;��<�d=7L]�F�<E����4�=�wD�yd[=��(=%b�=��<_�%�S:�=f�;Ӣ�=��Ľa�>�@=���=�Q<��=xb�:��=��=P�c�T[�]A������=x=ϻ�'t�=����\>�j�=`�x��B=���=?�=�k�=~��=�OD�)��<K����>=�]C=���=�l���]��'�<�x6>��=k^��<�T������Z��\�=�ϵ=�/=�9���<U8ʽ��!<b��T>�=��=)AF��d�;��/�C�=�`#��M<�c<Ȅཀ��=�r��ip>@I=�*�=�87�������	Ѡ=�b=��=�=�=�>�v��9L�NJN=���71ܽ��,<�ߡ=,��=���<]�!�z^G<�J�<���=�=��=�k�=	9�=�ψ=�� L�����%�";�c�;��=�䢼�p��-��;���=��>]믽��=t��T��K�A>���=������>����D'��=�l=�2�=��ӽ"���{�=:j��NX��j��=�<z�8=X(=��<��,=���={i<�H�<�W=�1�{�T=2��<6�=���a��H=��ݽOսpl�='��:��>Yh�=���=S0>�T��?�Y=�'�=+lݼW�7=��<��#>��[�j�+=�d�	�T�V㌼�Jj��j5=�$l=�FĽ�=<K���h�=�.�=8���2�=�#�p�н��=|��<c�ϼAX���F=3콕�#��b:�W��=���=]��=�����=�/�=��h��Ò��z���載G
>p�ݽc���C���^�u��rżH��ߜҼ�亽��S�w�����q�t�r=��4� *	>\)��q��nl\=�M��=Ub�;���#��=>�ռ	���xת�MV�<	�L< 賽�X��O���v��=�ɍ<�rg�p�>��6� ��Y8�=䠇���V�X<5��w�=$7-=���q >n��������$�Q��<�o��nS<"����ф=�5Ľ�d;=��$=�C>`5�=�>�;yo�=�Zǽ�o���׼�  ���U=CD=B!=0��=	E����=�y��H>B�=�d�=Q�>��=�4�>@$;>gr���P>�cȼe�G�v�9>�Ǽ���= ���:>� _>��3>��<�'=>�r�=n��>��=g͑��u����ؼ�#��ҕ�=.�H=^L�;�ĽQv9<8��>7=D'>s�<4$[=���=ǭR=>'g��MP=����3=��<��<�c�<i�>�'�_c�=M�E=���=B��="�[<�c=}-�=l<�=�h�=�烼X9�=�1E�Ԏ�<��5�R�=��ۼ��$=�H�<M��<�,=�ڝ���Rp<��I=��=P�Ͻ��{=Nk���˽=�KR����s�6�t˭=,�Z�]�<�d�<��=��=v	W=O/�\����	����e3=zH�=�Y�=�*K=�=B�F>u�=����=���=}ֆ<&-�=I�=z�<>/�켬�<[�� 4b;ᮟ�B�|<�5�=Seֽ�3ҽ�ۛ=��>�a���ɻ�T
>�HW�Dm�=�c���=��S�������	���=����	��E���=�;v���ă��ԅ=68���"�E��<�O�<l��=�q=ͅ���#�=��ü��
��{T<g�<P�S���= H=R��<:�g=���?�<:N�ʎ>���������<޹V�N�==���A">2�Y���w)='Bǽ�.J=o"�=��=Z�=[�y�U�7=𭜼���=%%�=�l<e# >�>��*.��F���碽�"�=�L�=��B�Y@�=ƥ�h�<�<��=��<�2�?ı<��f�k�&���<��={���_L��x�&=1!.=����'��=X�>b��=o�X�<�|=g�=��j��=W��(�=�I]<:!	<�~��M�nS,�ڼ�;��=ʋn=s2E=k�<g���/K�R�e<K_�;I�=���:P��a0�Bh)�SB��U7�<b�N�(�r;�I5=L8�IT=�!*������@��b'=R�t=Ԋ�=o:�;�ڼ��!<_֣=��r��"s=RK�=,!���W�=r'�<���=�ɱ��H=];<�C=N�E=`��/�S��c����=$N޼�Y�<m\f<�P[��s'>!욼n�=���=��2=��)>�k���%�=<1=!/�=5�¼��Y:�@$�@S�=o�=F, >�v%��岽^Ϸ=9Hj=5��=��$�p=y�(=����>뎐=�p\����=��<�j��	m=��9=-w�=�,�<�ŉ��w���懽��>r�Ľ�W�=�9^xU�k�C=�� <Q��N*=�m���ż�$q�Q
��p�<zwս���=��Ӽx��� o�L^�=�7�<�= �]���c�< q�2d=tٜ�� g�$�����ܺ��<�8�Ґ�;�!?=;z<�z�bw�<2!0��E��ѱ��I�G=
>��ぽ?�ý[B�<�R|=V��=�<RxX��㼅>h:õI=�F;h��th�=&��=X�u��J�==Yb�V#�r�b;M^�<�L��k��=�* =�`�=ݴ�e���3O�=V4>ӫ����R=�b����<�4l<�{=rn�<ID�=vS<Ҝ3=۫�=�h1���.<%1��|�Ѽ6w�<#�<�b<��<�}&�k�3<$�<`n��^�=!����=BC�=��^=wo=^@5�&�=���1�n<�1=�%���,�<%�<���L�=�WR���4����;jk�%�
�9�=����[O<쑨= e�����sI����=��=��=I@|��gԼ�|=��I���=C|�<#��<��<�f�D��=�L*��a�=��V;	v�<xH:=���=9܃=�:�=]VV=�1:=]A <����)j<�8�<����<�=� 2�ͼ>o~>�9ѻ?�x�uR�=�-�=��<<���=o�=uNO=�\�=~�=D�����*>~���*i�=Q��<P��<�UI�"�=��>�?
��R>�I=����==!�u=�iL�<�;�KB<+q�m��;�H��B�>�>�=>�����`t<'A�=�w�<��U=)�=�e�P�9=;{~�Ӷ<PO>�0�=t-����_=�"���x�ٿ�<kx�<�Ի~�|�(��<C2μ�i=����2�e���Zhu�Ù%<��?<��o��Լ�=�87��|��rMȽ�D�<Y-=�FǼV;��tG,���<%�ռDx<���;�v�=��?�H]<��������~:\��< ��5[���D=�I�:Z��+��_w��b6� 6=x��BWG�PO}�֚_��蚽?�-��$=�+#=���6a��R�<Ƞ=�a��z�N=w��<_��ఞ��N��F39]L=_3$��K�<�F�<��;�~��iF�;�p����N�>�b��<�8��,C=G�=ɑ7=�Z��#k}<�z��-���%q�<�Z�<��<d���j}� �S=/<�8�=�����T=c~e���<8S��c�<��<�=���ѵ<���S����=����@8y=(��������8�;~���Q<'�:���\��:���)�I�3<C��j���8P<�N�yI���O�;�H���<�X=��v<��=G�<_�=�� �=4=��)=m)��?�=\rF���ٽ���<���{.z=�(3=*��=���=��>L�(<��E=U�=�˫=vO=���=t:";������t�=r��0�>`b�?�,=|8=cq��هI=e�4��Ȟ<��V�TS������g뼌�>���}�<��<"?<��0�G��l�=���=k�=()���V������S���>!��<�m��y��Xa�]����k�=Uꂽ�k>x���=g��<Ѿ��"=��>%��=c)8��>�<�=E}Ӽ���=�И��
��q_����.�=����k~<"�y���<Δb�F �<�^=d�=���4⼌׃�>La<���=�5�3>ͼU7�Z*���=ɒ�>�"�=�����ͽ�^�<��C>��D>U&�d� �Z��m�<���2@���>a0>;�Ƽ����Y=Ɍn���佬Sx�rIڽߵ{��N2�7
�<��=���=U@�<��!=I,$=��=UE=b�=��c=�$�<dp=�Y;=��O=.�j�4�<�q%�N��P�Y=�w�=��޼0!��DK�<w�=NR<�n;�[8=�=4�,=�O�=}]���<�N~�ƒ���1�<�  �PU`�: =�S�=&�=�1z�S��=I1���b�<��b��l(;ꀕ;���:��B��+���w��I�=d�#=v?y���=��h�J��=<�=�eܼr�����z� =�!�=ijg=�_��p=y�D<tF���͵=q�f���x<�-<#�>aK�%$��U=�'��7>�2�1h=�h>��Ž�߼�\#��Z�}�=�|Z�憵�Sa4���	�� ]<�6�<��7̀н�ݘ=ʱ�S�=�P�=r�)=�E���ý��=)�D=��<S�ս��Q>�:���>�W>�K3��2��ay�<tn�=��=�,�=�t�����=���ij��5T����=T�
>�e�<�,:BL�=�@
�1�='�f�i7�;	ż��Z;b�=8(ɽ�o#;F�3��:�=�բ=#��<m�;=aȽ/=v���s<�"v=�$���=w�:rE�<�n=������=�=*���Ӑ	>m����N�=�+=¸�=����}3����&齎NU�0,��U`<1��)$>��~<j��=-/U�K̈́=�Ȉ���'���>��<&_=D����=36
�O�(��;4]�=�U�=�1:�c����2%=y�=4}���ي=I2�=�8���=5oZ���?<n��;���t�<��R>�<���:~��DAN�s��=��,�'>&=���Zm�<a�q��A��	=�T=���;��n�k=��#����=�cɽ�,�<=��<�(=#�=T{�<�'6�{]=6a�R9Y=G��<�/�=�j\��=�K#<���<�K�=o$��Ti=x����[;=$�+��Ȅ�(����4��<`=>�;��;v�<��S��2�:t�"���pTɺ�V<�j�=T_=��<Q���-=�Uw=$	M=��ȼ��=�0d�M/:=;�@�aC�=&(�=c�<��=n7<�`�=%@���Բ;y#���jR����=��</bM�S��=�5>	��D
����=`l=F�0���M��G�=�ŭ=:n��D�FI���L>�t�<N�=03�<I�P=�=.=���=b.>Ӯ޽ �<*ò���O=\a�������ốN=z��<��E��r�;M����9H=;���>�;o���?/=Zo�<�� >��>,�|=�༴��<tp�5D�<<k=A�@=��=u��p+<�o>�D�=��=�P>�,��e'����=�!=�>��>rS�<��>�/����=v��L>3�=v>e�}�A+>��H=��E=Ͼ^��k$>�H�������G�!>oYI=�#h��K�<9K�=oWn=�J��x�>�|�=o�c=�A���~����8�l��=4*=�P$=+�K=�PX���ڽ5H�<�:����O=eB<��M>���;B,����k6�=�>K~��\�~��=!'	<8<[<�Ͻ��4>n�<R�=@��]2=�1=�ɛ:�HO=�]?���ٽ6�<=��޽�y�=��=g�=�i�<�HϽ�<= P�:��<��,=��i={L�=C@�=���<��ݽ��n=W�轕4�)�)>�#%>�ټPq;E��7W|�=(�->1�����<=��=�ȽE6�&�=c�P=-fƻ�>Hm��������<��=� ?=Ş��Kge��ɍ<@r= 5Ƚ�z<���=����>�9�דx�h��=�4�O�<�����7>Y�L����;��|=Ee�=��<�t�|��=���1~=A=�P4�c��=Ɨ:Ƿ��.M�x���ܚ=�k=�ל=|��=m��=M�q=ٗ=SZ:�$ν�>�<�I�z�J�=�=2�|<��=�]����=(>���=��=ą=ȹ���[���<@�����^��y4;擼zآ�cA6=i3?<9k+>$�;>��c=�଻��<�2�<�Mڽ��$=��g�A�t����=)�;�n<��=�U�;ݰ�<��P�	&J�|��<C�=�`����<b�:�Og�<8�=�y�=��.<ɔ����A���\=U�N=��=!�K�r�<C�Y<]�'<�-�=��ȼ��/��5�=p7<=r��D���cֽ��d��Zx=xN�5tt�d"=�Z��&U�=޳=��>,���;R�h,n=�r�=��:��;��<=��8=,��^�F�Z���'�<
�뼯�>��n=@К<��x��7�=8�/=���v�<�(�:X�<O;ȼ9<<[ǿ<��r=7ϑ�x���� =�.�;D,��(�=�C���H�x��d=���=�s��Jp�4/��d¼"w�;��� �<��y=��h��sO=8�8=��b=�
@����1�<&�=�e=T*�BRU�}$�K~=>��<�E�<����$����u;F‽뤢�M
t=�rM������.l<yȝ�=i�<_ۈ���U��+���[�=l���d;�W#�RJ���%�=�8.=u����޽�����r�<���!�����={�<ZT=ڣ6=<9�= ʉ=g�=r�$>(vv��e%=�8g�и��6�U=�eｒ�2=�q�f��=˿=�t��)�=3�J=(�:|g-=oܼ K�=��=l��<!�����<S����-~�ٿ���=}*9���&=�UJ=evj=/A>��<X�C��F���;���QE=q�=I��<3�9���=�Z�����4��]�">��*<|I�����bߐ=�<��m���-=�x�=$_����=&�w=VE�<�ɻ��~=P#>3�=�W�8�A=����F<�[����1�!HӼ���j�9
>h¼A�|�N�=�8�8Ͷ<�ʑ�I��=f{�=�\�=J`e=�Vj=E�<uW�<T�)<p��yFf������Ì>��<y�%;*@=`8���ph=/�W<���=Z߉�X	���"�<R�*��#�=�$%=�W�=�
9=sͦ=i8��#J}�8u�=D�>���=����gϽ�3�=գN=ZMA���=Լ�=q:]�b�`nH�S�<�t��NໍA2�<->�� ��k�=i'�<�W�=g�Z=k����]'=�l���� �<��b���>�C=��:=9 �1��c�=�#�=��l=���=�<�P�=b��Z�>�=i+��[	>Wi��f�;2�, Ҽ4Ay=b������= ��<���E<�<,L�h�K==�&b<vC���Km=S�=��k��k�����=I�`=�����Bk�ގ���(���%T��ˎ<[}{�Z�<-`���%��V�ę�;E�<�*�<�G�<���=6ǵ=���=0�=^�=/� �}F��S�s�����	=f�d��=21�=}T�=Q��<�v�=�=�x\=��~<ɑ�=L'>�M	>��D;l[�=t&�}�<�0<�Q}>%_=�?�=J�=$�}�<`�=�:�=��9>4����=�˾=+ 轖�=��׼�[�<�vD=�<׌�$;��܂�cޞ=�>�h<�`M <	Q=f�s=���>�< $�=.��R�>�"
��.��J8�:h1�=�ǽ���;��%���>��̥p�3�>\q8=FY>t�><�>����=�]�=u��=�D(=㪺��g<��m��E6>�+>x�>�yC�c̪=��>�$>7 >�v����=~���Kq��>�I���k����=t,>#�p<y[�=���=�<��亿C=H�ٽ?M���zT�p�p��S�=��ļ.n=��=��������� �u�RL>$G�==Ƽ�= T�=��>�8�:��;ү|<��=�)�=�#����=_�<#h<��Y=-��<d�9M=T���<x�$=EvN�_@���F�<�6˼��/��&��y/=g�<��/=#��=���9c�-=_'=���<�<���="���W��;d�;z����m��%b �?L��Ԫ��\t��]=Hܐ<������ŹSE�R���t�2<�0o=/\�¯x�
���=�Q"=Hel��Ռ=%�=��� =���<Hh�<�^<=�*j��V,��כ���_�"�#�Xr=�ػw�=�F𼚢��$1����k>�`>�Et>h>�O�<��B>�B>���;.>{��<��'����>�Ѱ��i�>j5T>�}>ېe;C�=f��=�OP>6;�=�S ��Zx��&;���`�=��=m��<����_�=��=��;��=<L�;�a=��#=$�=�>�<)�6=/.��O'�=�"=����Wp�<�ؘ=�ֽ\��=7=��=>��%=~C�=��;=�m> ��;@
�˙8�vh	<��G�a+�=���<=�V��8y�{Ѥ���E=���=��i���Ƚ���<�ܨ<��`���R���0;J�<*2��8���׎=f�=�Ѧ<!�(<�=��K�<D ���9�<��==Lνǿ�=$=�y��3�<Q��*�-��}�����<�A�<k%�=|�et_<�Zc����;N�Z=�6�;���܈*=KF�X��<ݖ�=�<u|<D઼��N<�g�%㫽�h;�*7�<19��@4�֋h<��w=n*罕�P<A^=@��<�9�#����u��`���s��v�=Cɽ%jT�Z@"��P�=�����2�z�]�:������r=��=no�P#��������꼺E�<M��=�8� yh�f�=HNX���*<��">���ͻ�����;;��<�8���Ќ=����p�r��s��:�=��A=�|~=�z<<=ʽ3�߽���<���=�d�E�[<�/�=��d<�Z��?F�W�W��3>�A1<��=�};�ko���ϼ��
=\���V�=��< ��=��2���<q�����e=5������>�������`�7yH�����v\�<�X�=lB>z�?��R�=�@ؼ�nս��>r �=C�'>q�=��==�ѽ��D��`����ʻ07�=��=�}3�?'=<�=&�(>�@U>��+�x�<�M=1�����=cR�=�k<,m�=�'<Ѽ=��У<8�=�Z�=5�<{����N`k<�)=���Ƿ;��>�'�<��X=.*�Z\�<lF'��%h=����P��=�;$�b/�--5��y=d�=�D<�S=�g%�R��=��<��ؽ��o=��==6x�<��>�a>�1,ʺ�}�<;N�=y�!>��@����=4�>" f=�x�"�C�e�a����>�5<���=13U��4��N�=�0�=��V=�Q'�)z�=}A��r��=�G�={�Y=Y�'=���<���}M��P�@=1�=��Q=&Oӽ�2�<K�����;���;���<�ɳ<i�qm=�#x����67=��r�j����`<�k�<hQ_�K�~��J��Ż����<��A���>�=�ē�7�<�,�$K��![�U╽Q_>��<J�d�eE�=�����U<s�M����<$Ͻ�w
=�ü�^1��\>P�>�b�O�ؼ��	�S�������<��7�</TG;5#��/U�=� �t�;�/����� �@��=C(Z��]==�:Ž���&�=E�<�2�<�A1=f#>|*��b���]T=<�M�=N�T�\=i�=���<�j��I�/=��i�dDW�ه�=�2=�%�:�q�<0�;����M��X!�=0�D��7���3�3�#�a���@�T��ܿ��26�-/�	�<���<#@d=����\,�����KV��Ļ����!�"�Ib���;��|��/;��i=M�D=_���u�+��n���F�=Vc�=�ݺ��������j�<z+��[�<i�>Dz��Ż >�  = ۝�yb��*�(�b{轲Q�=�p���!=9kɽb�=��I;�@c=�26��e�=x%��2Z=���<1>�ԏ=q诺�<i�½a~0;m�=a[�u��<���=܄v<�Z>�Z>�`�;Q��<V&=e��=OU�=��=t�=i�=""�<=:M�,>���e�AqI�0��=�pܼ�A��s�vkF=�'�=�<@;<�=���<�tL�m��=��=�q=����ޞ=�M�<}�⼔*����=H\�<xR����\���?=U$�=�l��O�cm�=�Í�+>n+����=��[=���=Z
���d�=�ͫ�R�y=3����=Q�1�j�u=>�a=��Y��K7��<�=��=�<�e���y<���<��K��V�<0��=(�*<z��<8E�=xE�<+�[=eO=�<p� <'�n<�/��l��zp�=��<���x<�z=l!�=\��=٪���{O�.��=��|�p#�<Tl�=).v<$Y6�!
m�aǻ�Cּ��x�"�=�=�F�]!��'�<B>Z=pV��ӽ0���z{A�%�<��I=OL�A\{=hZ[=�:�]Տ=F����=���=%�=P�:�6�<�����3+���=�m8;���j
>�7��g��ߨ�<Դ=�?=���=Sf`=��=��;��f=_�=B{=u� �S=��P���=�Sy�	��=������ν��*=��=��=]�佫�ҼO͚�e�ʽ8���"<>YM�=�YP<u�?�����\<o=<A*W>�1�=Y>V=�XսQ����kH=��=3���|o=\R˽�c�<9	ڼtD�����;>n�q����ﯽ�fȼȑ��FwӼ�j��K��J�&<jDh<x��=Y"u=;K˽ӵ�=Zfo�x���罽`� ����=���koʽ}�
=�����s�Vf�<In�v:�<�{�<S3� Ώ=�9�=�O���5#��I$��䶼�*׽��׽��p�r^=*UR���D</�>�SB�`%�K4L��v=Q0Y�D�=��L=(������:Hƽ�<�=�#�<��=������b>�H�<5�a�]���ْܽb�B�=I�4������e�5v=|�5�Q�3>|�Q=W�=�=�8_�A�U=�]%�����e;՜��=�<��=k%j����=(K�U����x���U=��=1�S=���=�
x<�b�<¬w����Sw��9�{�+͙;.���7��=��;��<=9���}��=Oyͽ�����:=�(M���=���=(nU=�~�=��<�]��̲��O�>�=%�|< `a���.��c!<�o ��"�U�=� ڼ�_=�+G<w�=���<i�'= ��@�=/g=Vg3�D*�<���=�����h<xt3�@x��·�= v�=���w=S��;K)�=�e�=b"�=)�=�B�=^׼U��=�=Kb�)��=O�ݺ:�G͌<����C>*ש=u)h=�5����<>�;�e=">>����4�0< ��<?�;��޳=���=��J=O[�<$��<c�Z�џ�R��<��=a$=�E��c���?<]Q�=���=�qE���=�[���=1���R>�9��
��=q�f�e�I=3��<�b=�0 �� �=x��=*����z=*�u���,����=�p�<�t<v�=Lzo�ZQ=����fb�<��1=c*����=�$=���=�B
=��,�N*8�ω=�7�<+i���x=��=��a=�#=2O���=?">k+����<��=��T�F�(t�=�No=��'=8�<���=��-�����C �<O�=s��=�ʪ�|t����=��2�;Kk="ak<��%�=N	ԼB��=˗���xx=��2;z��=x4׼���=�Һ�̻=SF<B�f�L�< ����=;�)==H�����=O�W�TrP=�k�<c���:�`^���=��<�a<�[=��>=��i=|�s������nd=��=���=�2a��=8<��=Ij��5>�{���š=p�=r�?����<�$>��ݼf�<c>�Pֽu�=���s=�1>�,�B,��d���Ư<;{v�hg���6=���<t7�<7�==�_<".y<W�-���	�����p>�������Vս5�k�m�<�͆��3-=x�μ��@���Ѽ3'c��6&=c�p�'�O�]+߼J�ν"=Mz��A���@`�p��<�W<�ۮ�C<j	b��^ɽ��-��>O�(=[��=�ؼ	W#=�^�=p�c=U<������C�*=We�ZW&=�^�;�2���K �7��^�3��ٰ�h�３0�;�5��%{I������(�<eg��"�m�J=����6[�<C=_T<�N�;h�=c�<�Alk=;.<;�e<�<���N�=�3�<�'�=b��=R/<���< ���%�.TU=�c���a=S�<%@=Z\D=��<�[7B=pK�=ҁ�<�&F=ǝ=���=H�ǻƍ�<U���[�]�{�r��TG����=�I�<qy�<]��=�d=~�<*�x=,��<�mb<<]<)�F���";���aW�<{.�=��b<��<'%<� ����<6Id�PAֽ���y=A%�=��3�B =�2T=�$>=��U
��"�<����#W@;)�¼�3==�P<S�&=o��<�)��%2����[�=v�.���<�̼�[<��	�ig!=�*�87&���<j��=7�����=W#=@p�k$�����=����<WF*<���F�I<,�B<��;c~n=��
�`�+�˭�<�G�={�=�����O=��4=*��=0�=�ܧ;�(F;�9�=�S���<�[޽���<�I��~.���=���[l<봀����<?��<��0��e=w�<���=�|�i�<�:�<�������Kw<I�ü+Mo=`p�=t�S��<Li�=�@f��=�W(>�2�<�:=�=4+<�H����u�٩�=�x�=,�=��>ˆ=ۂ=Y�����=X�J�k4�=x�ǽ���<�^:Vy����l�I��=�n ��t�=�(�<'=�ǲ���N���:�>=�=Yi�<�|�<���=�e�=O�<�T=1�=�$�;����6Q>9;��ƽA��=dE�=J�}=�x�=U�x��T=����ם=c|�<��D=�[Z<�>N��=��(>�gr>|<�����;�?��U��=<Q�<&����V<<%Ӄ<�����=W�2�<m|=EbD=� l=`?�=���=e�D>��>0����<=��=N��1���iF�=�W�<$��hzJ�t&>�]�<�r>�=����k5�=:ǐ��}+=;5�=4Y�=�'�0�<u`��
7 �����<�=���=��z#�=�9�=��G=��е=����3��=Q@�=���8=��d���4�������������2;B�,=P'U7�?�=�T<���=8<>�u�����=����#�	>li�<4q�<�Y�=xRq��@�=�e�=�	�kaӼ�=�劽Ux�<�(=��ʽ�1W=�%ټ��D=��s;��i<�Z���:݃=��O�[�`���`=�=�%�=�W>#�<�7=x1�=�^�<$ܿ<FV�_O���3B�QS�<l��=���<�ϗ=+�K=[���&>�P�<X��=��=b�d�?e���^��н<�*�a�{[�WY�r��s*ս4�������b=DŽ�q����;�GNc�_J;֓��+����lM���>����*�Z�佒F��a��}��M���ļ�M�={Y�"]�=���=+� ��Ľ��)=	�ýǀ����y��g�<M����o;�6Ž�����
�?������<@�C���ȓ�=\�#=��߼���	��ȅ�=����e��%;��� I� e�������a�����B���<��v�nmm<b�9��[=�~�=�$P��AH�[+A=���=�l�=���=��"=��g��-5�!�<m��=�1=FJ�=���=[c����8=���>��<�z&=8��=
q>=��>=���=�{c�cYM�?����-�=.��P��=hN6=2���:t=p�T=f}��S��d�U<�<J�~=�E������5�4=k=q�EU��R�=�3\:?�Z=�'���<�5=]�˽���;����O��>��Tٍ��伿"߼�>�Y���sּ\<���p�r�۽Jtܽ	���ǽs�S�ˎ�M�<jP<=p�f�=o��+E��=��;�h��d��#�н�D�Mg��R1Խ`�������8�������1F�;.ҝ�
ji��Xe�:=v����귽�¼���<�������F	�ߧ��4���=":��� P;��P��w����<H�Ѽ�n����>����a���/���;��g�C|�6�P�\��%�= ?h��H�ܨ�=�ɼ���<�L�=�dG���B��<=n���#�jЄ=v;=��=���=_�,6��OB�=�)��k;#��%=�(<7�a�
B�<�]6�7�5�P8G<���;;%=w/=�{���F����=���=���<�+6=�̈́��T�= �=��<��9�<oz����P<+�^��җ<� b=�D�®<{����̩�F�=�_�=��5�i�<�9@���ͻ7�ּl�7�X�׼��Z��;u��<O����
��w�=��r�.˙;j�]=u~�=�l=��=�*�<�q�=�
�=�o���`<=�����x1����=��$<#6�=a����=b90>�r��{[�s�t=\��=JU>@މ<uߖ�1=�
��
Q<U
=�ʀ<�j�(�=Q�=L����=��>+t�=� ���ֻ��&�=^*7=�d=}�Z��Kf<�|V=��R=$;k=+Ɔ=[���o���<���<ѕ߼oS�<��&<�K�=_7�=��B=�B=³�=�����(�yv�;Ŀ�<�v���$���\=��$�1v=J`$=&�<��=��Z��=��!>�_�=I���p��xr�=�
'>2m<|�==Z<m�@<ڪ$=B�>ܽ#�t�e=��B=\��<�0=(�ɽ7�=�ә�]�-<L/�=�Lo=cp����=r_�=�z�=��ݼ�U=i�����=��i=�Ȯ�9�:Y^�=��=t!���ۼ��</S����o;�"�=�ֈ��)Լ�:^�5��=���=+X�<#�>���<�]��pw=t���fs<	�:l����S>�X��CQ�=��B=M$�<sc<S�<_��<9��=�ή�+=��_�M���ܝ=�*��0)��b���E�=��<��<�L6;�\>7b�<����nj=G����P���D��f���<~�k<�c�:���=�wv=��?<�#��u6>�1�;h5=�Ҹ=���=��[�ya�=.$�='�=��q���������>���\�<���F=��>(��=���<&��<:�=L��=�mf<ؐ�=X+d>'�'="��>�}»���="�*���=�i�=�N�>%z>�8�f2?>�{r=a�̼N��=\:����= ˤ='�=A�>~����=L�>�i=�.>�,�=�(>��e>B�=w��up���m���ܒ�W�(>���=���=:,�=���=�}>�>C�N=�!:=��+���ؽ��=#>�=�'?=}==6cO=xg=s���=�<`�J>br;=2f+�Q��=o�=h{�=/ZȽ��*=3��=�	f=/	X=+ ���1S>�%<��>{a��L>"+�=�TD>��Q<C��>.~�>R��=B�>�S�=�8P<c�,>�u�y��;�ї=INR<;z>�����>a�B>�=���='O>�o_>�4>O⪻�d��{F�:���<U�-�p��=|�N>�;���;��=%UL>��>a��=��"<O4�=���<^F�>��V>-�Ž�Ȼ9�<��սvֈ�|�C���>�n�=+>����>��<.И=��8�L��=�G�=�J>8[B>t|�8��<9�=]T�d#�=wϳ�c���j`+=�ȼ�������B;<��=�p���eL��v=f5)�;hH<�=hEe�di�����:kB)���e<_i���7���=kq�<J�8"�=G�R<�Y ={�μ^4I�����!=���<S��B��ut�;�џ���>u �:�fo����:ȉ=��¼C�#��Ԕ��s=Z��[Q<]��<�Cɽv��A̼p
=?��;p��<��.=k�,=_>,�+�z���1���o�*����<�~�<誼9 D=�G�<Rl5���=R�9=��/=�<�i�;��u>˳�� r%>ϛ���;L���@<��=�� �*��h�<���;�,�=Z�;��=_�ּ4�^<�|=��>������J�='����j=�3E<�_�=1���/�=�%h���=��=���<�1L=�<A�";�<�b[�>�D�/�<]�f<�����ļ=�8b=�N���L�=a��=8+��B>��
=j?&�&�)>���<���>Z�=�K��h�r=��2�в<�*���=�:l=���=��D�����<����9켓Ih=�н����R5=4�<Vq>r�R��,p=b��=)�>5�K=~X]=�Q�=�R=���=��"��r����%=�ݕ;9���G-(=ߓ2=0�<&ۈ9�� <���=iQ�=d#�h��=Q�t=B�<�9l�� �=���<�| ;`q�=`�=/Q�;�%g�ŭ�;�q����g=�!��`g��'�<���=��e=����0�=�7��m>�Y���=v:o=>p�=Q�V=�=8�<@�����=m�==<�)�)5=����a�	=�2�=�=f��=�N�g��g#�=,��=FE1<�;=�ߣ���=�D=�ۼ��=���<����*L>�>Ƭ��Z�=B]�;6�c�Z�"=&Ƽ��̽���=P �O�(�k�=���<�:#�>�.=D<=g���A=�w�= �=�|�<�c�=t>��>hO�v�U=-�=���;�>�zj���	=��E<�l�=�ta<��Ѽ��=�->��=���<�+̼ �f�"|Իi�0>0*�7S�������z<��1=���= oA=қ�ˡ=(�\;{Un=X�=��ӻ@��=
%=����a�2F<��û#a�=��3��f���l��!<T�>E�>�n9=b1�=/��<#��;�k�<��{=�g|:\p�;��=&�=򿁽.�;�C黅������=n����c�y\�=ۊ<�j=M�Ƽǉb=��&��_>��,��5��fX�:i����@6�p�(�!��<<�Լ�4Ͻ��M�k�g��u�<hϽw��hR�=�$t���*<ȯ�ȴ��ü�<Q�����*=^~	�>�v�qdݽ�Lv���p+���oJm=��D�ļ�>��=޸м�h�F^.�1 ����e��������=�M��Ռ���@��CH������;m���2��ýu�� -C={�_�J����+���=���<��o��:�銽����"�8GL��1ü��F�/6;͖	�q9��=�.Q�� ���=la>���=Y.>Xlm�F�=�?p���'>�|$��@>!}*���;c����<>.�>��=>_.$��H>�<>�Z�%6t<��3�_8�;�bͼ�U>��=е�������>���=�>=��ý���>Ůd>f�>�^�>I��n<t��<�oU>�m6����M����=7�����;�7��/�>��>�g�=ޓ�>��<�]�=|w�>?j�=�i=Ƀ��ݒ�J]ͽ��]gd=Y/�;�U<X�=
���y<��<9=c~�=���=�'B=Q'<�h�#���e=��>���=u�=�\|��,�=��=��<�S�<��Z<���;/RI�W&+<�&������٣='�:�4=+��=50=�و�5��=5c�=���<.�=a��<h��<Q��<�i
>s==xV.>F[=�㗼w�ݼ�ms��C<�#>�X�-�=�>`=3����=���;�^l=�J�=���=��M��yL=��?�0y�<r��<�v=<��S�	=�����=a�<�E�(V4=BX<��ɼXpL=�NK=�̈́=i��=�=��=e�=�s@6�h��<�;�q�;���=u=��=��<�5�=��<�2�<�j�����<1Y����;��Ž�������u�=�L��½|J�=N"R��滈n>Ԓ8�Ħ���m=�ɷ=
zu=���:bF�;��=A�;����]=J�R�����Ul=�kV�*�������誽�����<FԼ�H�1Q���û��ѽ����<Vn�<����s��<A[��ȃ����k��2ʼ��z���g<�/�w�=���=!��=������=]<�;F�̽�=��"�?4����<�䡼�=W��<�j��L^�=H�<'˂�s^����<��=P�!�M�=�;�y�=��<�L�<d�<�ۼ��N��B���<Q�=�8����<'=�S���R���*涽(+��u�<�?���li<�=t��=�	=0N�;��<g��=y#��T	����=��>𽼹I-=kAA>���<�S�=-A�����<����~{�f��=���=�.L��i=hx�<��=��l=`�;P��<�'�=��;=�AI=ӗ�<LV,���׽�9q=29�=����:��<�k�=>�=l^5>��g<l2N���Ƽ<b@=:}�=�����ƻ��)�)>��>�uE�|�v�2��<�m=<RG�9ɪ��[��e}� c^�u��=ä�=������=�zF={Y<��?==���ry�<�ټ����޳=7�7;��=�/��[�������N��e���8=
S=*�I��D�����cBa���Ӽ�<G�*<���=�g�������<3O�=o�8;����~:�<YB��o�=�a��c��;��=0��=� �<�\(=׏μwC<|���<�F��������;����=�o=��ܞ���c=R%���1=����(m�ON=<��=�H��Ӂ��gt���;R,����=���=fG�=a*�.1=v�x�'\�<�R=v<�g�=#�H��9�	��<W�=�N���n=L �<eu6>�>��<���I-�<L4 �p�c<T��= s>\�=z0m=���=��=�/�C�=#�?=&�3<�4�=��Dv�Q9�=�"b=Ĭw;���=�� �4(=�ڈ=nŉ=�{e=89<���=E�>�w�;��>=�����=@s$=@�,��`���G�=��z��=1��<a�%=�<�<t�-<��=�R���%޻��м�F=�M����=1����\=���<��\=!�1=W$�K���t��P؝�y-K=��;��>�K=P�j�{#�=�a�=���<Bx=8'�=��<��<����Q^�M��<�=jI��O�P=�`���L�=�v���=�ߡ=d�v=�
A<�~=�a�=G��9��<=J��=	=$$�=	�<��F���%>�O�=B��#��$`�<������sqJ=���=�/=��8+=ݼi����L�;bi��4�jn=��	}����3=ȫH=_��;=!'=�`C<�oڽ�Չ<�U�< ;���&���v=[� >�����RA=dмl�����=`����Z_�{�a�b,����Q<�w9��jp�M6�<ؕ�r4g=zɼ��T=f�,=k(�=�ټ�/�;X4����=�=�]=ׯ�:U���l����CC=w]<�wt��T���<�J�;.x���%=8��<�J\����=Hμݗ��=d�=ʏ��w����><�{�bޔ�7Z��sG=���z#s�f<�3D�+��u�=O)�]��<����,$=�tD��D>�ʢ�=�G;��F�\.=�}���j<��<m�~�S�<#Od�).<Q��=� ?>���= -����K�7]��#N���������=dY=���<M���x�<4Cļ�	C=��K=�	D�� �<)̦�=+!=?}D�{B�=;8�;�^��U�1���<�<ͽsT�=�h�Y%<5��W����e�t .��{н�W�<I˕��:�<t-#<T��=�s3���=���=�:�=�>���E��*4�v.����<�{�=-P��E�=�9(�(Y_=�l�Tg�������ѺVF�;�<��P����< ��=@I�<!���Ѧ�<�J��4��y#=|F�<�������=�ۛ=�ݳ<�t����k=N�.�G�I��?�=�,�<y�==Z�=ղ=�5���y~=�:�=T9I1-�0轳�<�K��;'�5�9��=e&����[�q6׼J.�=�m��w�=�!{�/�>�}�B�>T >w3�=E�>L�
�m�>��>1���Z>sܤ<��$>�*>�3�<�A>��/�}>��P>q�j>�ʊ=�]J>�<<>*Sa>G�g> �$��<��;=K�l<�p}>��`=��1��=�"�=#~>\���>�L>�5->@�/=�!/>!R�=���<�F>hs[�,�=;G���=�:����&��<�G�=��3�$�.>m#1>��>Q�0>���=�h>(�>̉���Bj;��< �M=k O��.�;�L�=+^Q<�E{=X�y<�!;~��<�%,� �c=�%�>b�<��Q�;�z��V��=��<�Uʽ���Q�4=�+=f�=bF�=�:����>AQ���u�W�(=����b7m��η�M��<�� >��m=4�G=X��=5'=�
}=������>U�=�
A<�q>Me�=Qт=]		��^�=��<�XͽR� =0�ؼ�k�<22b<���<�\�=��>�u�<��>˸,=�0?<���=&>�=��=TY��E�7=uD<���=>����򚩽6۫�]���~�R�L$���+��ZxT=��<���P
�=�_=��^��Ex� �6���P>�6��,)n��!�v���: ����֨�y$����=�Z4��V/=�>.���n�Q�[��K�ݽ.}��x��u1�=��m=���=�I����=L�*[K���ٻz�5��X��#�=��u=�H�=��ս�j
��F�=���=�����Ql��d�=�ۼ8����HE�`��<nA=�Y�<����V��<E�}��f�=�������=a�T=a�0�=`�<�>D=���=�=�u=�֠;�cܽ��n=J��=(V>��=ݝm�	+�=PI�<�R�=�=2���ռ!i�=���<>6��@f>z������t=��9K�<�V�=̲����=�Ƴ�ie>�>I��=���=��H<��=���=|�=,��=4�F���U���7�<=&D$>��<��O��]�=+�;�0#��h�"�!=c�<@D�<��<�e�=d�b=��;����5[�<%q*��"< x��<K��@���o����<^�)��ݓ<_z=_.��՝u=]>9�<��=�~��j&=3Cg=�"�=�U�7�Ӽ
�;/�= �<7&�<�;��<u]ڼW;=��< �»�l�<iE�=&JJ=�Ֆ=:�<���oq��,����֕<��<{��='ٹ<kĺ���<=	���c�;ͅ��Z�b\T���> 	���z��:G�H�=S�b��4��M�=��*�/=g-��c�=axT=s�����=0�<��?��'�=_;�=#<=��O��Ĭ��S� ��"#�=6��=Ύ=�I7<�]p��=�=Tٞ=�{�=�a�<f��<x��=��=���<3����2���u�1X�Ce=����z
�<�=��=�d�=]n��I�[=��=���?�=��=gKI����=�Px=H�f���=O��N	�=0F=Tx�<�8���h>r��<�'�̍Q<���=^K�;��>�&=�����_ؼ͗4���Y��J��K��=��W<�ٛ=j�7��S#>�ߡ<J w�߸�>�0�0�Y<eԂ����=��5=�QA���Q=l�>߂D�g� =aƪ��>�����x��<ܰ^<g�t��L�=dd����=.��=�/�=󯽘e�N->�佗t<���w��>�">�'�=�^�>��A�+௼���<j�>p�=���^���y���=`��P����M�<�>�2>�*�<s�>��?<����Y�>�r�K["�3^�=q���X��<M��'��6Լ{��=���;���=G��'À3=��a�)Ҋ=d^�=��2��1�:�=�P0=�vT<��;� �Z=�W|=��<=��=��= �==�5=�짽���<�s<�Z��V�'=^�7=
\�<&^i�Zc��tT=�ꟽ��=����*1�����Pm�<���<�Y�=�C�<���=t�ȼ�F�<䮐:�
�<&
�X#
�:|=��<�Y����,�����BL%>��$�?Ӎ=���<@z�<i51=��=�=����<��p<&_Z=�%F=I<�=��，����r�>	����E�<>�g�HC�<Cs
>��!>��=�/6�l�}==>jG��}Vܼ��*=�4�=^��=��=�e���'�����=&�F�~�=��#��਽ȥw�!4w=�zH=�Lk=�=|�a<r��<��C;
���ZP=d��;;d�=���=�`?�
C#�a��&=���k�=�B<�"<��=A]�;��7=�{=l<H���@ <8�l=M��e2x=�=&<�5н���<L궻fqý��=^�=��=\>�=^�<�t�=_}��h�<����><�t=B�=�1�=���=ak=��0=��{=p�<�6�<v�C=!L½Zs㼨_x<]�>F�t��n=�}����=)�=�6�:����%:�=|.�=$L�<d}�=�b=J��<���=Ѕ�=��;rԫ<��۽k�<- ����=���<Y�> Q�=$�=X��=�����[��T�;@��=�s9�����4<�p�;�A�<�I����x������c�;w�����w=x輳���z����=?4���!�<"���@�<=����y=(~�Ē���x�<�=ށ�5w'=��=����-�F�қ�<�!�=hVa��s���܈��¼X��;U�<�c�������nD=�B���w�S����-X�|��Ҵֻ4�<�|P�2g�<+�B=i ,=H�=o�+�
Ӳ:���<�X���E==�R=)�=��;�d�=���������=�3=o��=�ۺ�Yh��P���=�5w=�} <�$>�U�=�7�<�1<`�8;c�:=��=��E�z,p:�=͔<
�y���\=��<����Z=A��;��=�ڜ=C�<$Dý��<=zl8��礼p�R=��=����4�=!��=}>�����F��t\=}ٯ=Ӝ��:�=M]e=������ ���=���@*��楻f�=P�=���l��;cA�=��[=������=���=o��<���=F�L�K>�n��#�=��ؽ	
�=�����.;�\�=���=�Z�<� C�89=Ve=��up=��G���9`=D��=y�2�*b2=����o6=0�=j�=��=�,��S�u�>eÄ��B
�@g=��޽|XD���<��V=l�C��t��c��=���=���=�f�=1��=���=�{;O�;�"�g=%7<}6K=�z�o���V��<��=��=:둽��W_H=(�=#�C��r�={�=Z��qY=����B�ncD�5a��r���S�����`�B=�?P��
��t�=°?�<Н=y=�=�=,)<?�=̢=뤓<���=���=��E�C`�=ߥ>=�y0�tҞ<��V=�q�=O-�;�;�<�.�:y��(�<�#]=��<[
�=kG׼��_=��{=�i= ����>GP�=g��<�Z�=�X�<�O.�)첺��=���=+�d=�=��<�ؽ��<�7�_�O=ۘ	����=�=���Y]�=��<e_<�O�����:E=>���=J̑�X�ͽ�s����^;et�=�=�l=ф�<Q�<>/�EW9=���dt<z��;�w�<`O9=�zF�PH�=<M�=J��<�F�=�H7>��/��C'=�X=i���n ��;=N����=��=O�a=��<�=�'�B�=!����z�=�Rv=]C��Ϻ<�ݻ��5=�C>�ʵ=�`��#��=X_=>�=I^�=�&�;�a=��<��s=��p�׃
;���;)�;�]Z�h;�*t��}�P�<�΅���=����)\>1�=�f1=�?!>��r�+g9=��>�e��IX�=�I\�h;��U�U8ҽ|뛼�[>=z�<�:>�6�=�er�ך�=�p0<���軺=�t�.E|;}-���PF>r"�<0:�<E n=��K>��{=��C=̱��i1�>uYO>�d>e��>#n�B�Ƽ �<"�4>)WϽYνWL��~��C�k�'$H�ｎ�>s�=�!�=���>�$�]�j�J>��<H��
�"=|CF=zռ��=	#�_���_W�:��f��%�<�آ�/u���v=�!�=?���m��^�:��^ <���<Ӳ޼�~	�R��;�
4=�+-=�P=.�<�E=C�|����<b�=��������̤=��==٠2=����{<�����-�<r��<i��P��<~y<�����)���Ǻ��ҽ�\�����L��bi�����<�=PfO<��J=S#Z=a���K=�u=����_�4�'�N���=c9�=ⱑ=���<�՟=?�=|���+��;y�<�#>�U=��1=�>�=u�(<�	�=#|?����@�.>�bнLO�;�Ỡ`�=�󙼉3>��=�<OQ�=&3'��C>��[<=���Bi=3�������G��1A>�e�=	J<�\�<�a>��=^m >�Ҁ�R�>�>.��>H��>�n�=W�<`�=<�\>�h�<FfL��{��W~�<]�<����i���)�>CF�>�o�;��F>:��<��u��η>��;��ļ���Ģ_=#�<0�=��r�v�H�Ah���P�<Ż�v=�x�tv�����=�ݧ=���=#
;�t�R< �M�N�=��=��=�ާ=�+�9*��~�=1��<��6��="�:*x����8=U3N�2-�����=$mr�F���^�>6M���t9=6#;�n�<���=����ɼ��B�� R�� �+=g��=x
����.=9�;��y:���?��<���<`��<o-���=faQ��XK<�7L<����=�Z=���=�#���.�=H�L=G�=:��c=��<=l��=��o����=,��=P��
t4�N�=+�+�x��E�򮕼7�=�
���&=�#��G�h���<vy��Am�۩%=��=,	�!@̽Z��=�H��ݗ��Ӡ�=�$���O=�����=0>�L7��C?��b��fL&��n=��w=0_H�����392���=�@��FGJ�P��=3\�;>�=�������ۂ�_�s����=٤
=Yxýۺ�=P>y�^2�=�s=R�=]}��˸D< ������=DX����<�iC�CQ��b�<�Ѐ=1����=���<��=�V�=1��;t'?>&��X;=K ;��=����ԑ=���=�6���>��Z��3�=h��=Vu��9��=�1�<�ÿ=���=,8�H�H=R.�:g�̼��~=п�;�����e<D��<9�5>K[h=Ӱ�=b����$��ʐ�JU�4�<m��<��u�5C=�L~=��=���<>4���N<V=�/=��=�}���78=9W"<z:> .=.�<�b= U�;�����.�d��<y�<
gr=�̟=҃�,�!=1�v=#��<�!=-j����=L�=^�c=ݿ����)R>{-T<d�����E���>���C�<���� =Eػ|Z<L
�=C�=�$�<㊒=K��=���<�V>�'�=�=�=>�=�6=��=i�=�l=L�����]�=˞��I�<Nz�:�v=���;5�;����ʲ> �=�-��q+�b>W=Ù���+:��=�W%=װ3��*���\��-po=஼a���ˮܽdt��q�;��E�ܩ�Yu���2�<�	Ž��9��ӽ��߽]Q�Bz[�sf��~��	���6�н�Ӓ=IL�=��<J�!=k=��󔽵->�څ��O�#���q�>�8�Y��s�ܽdI���=���;{��{�<a�ֽ=QJ=���<�/�%�B���`;�A���뼭��z:@��;wj�=�$ڽ��ν���<_wX���(���=.���=MM���il��T�<z	����<����=�	�>;GX�
�>�`�qXR���
=TE�ī޼|����=�Լ�%=��E->k���<��=uw�;Dr9��#=4��;�{�<5c�Mi��-)=R�S'=0�нM4:>�̞�F�=��<>MP:�_�̽D*��2Z<�:<��;�U�<�b>=i<�=����	=R%>��>��m=�F&>�N5=��<7 �=M�3�V�=Z{�=�E�i�<P��Գ�=�Լ}%�=q��=�����D�<�����U�=��=N:9=�=�{>=�i�=���=^"�=��">Ow��%JY��^g=]j�<���=\��=��=��1=	g�����}Ǟ�z	>'��=r�s=!'=%eX=�<<y!�=ϥ�=G��=�=:	>���=�gJ= q�=��#<�� =��	=O\
>�fD=d���C�
3�<</k>�m�I��=p�4<��E���=&$�=�+�=��L=��F�/?���O�=A^b<�3�Mh-�[�	>IW����|<�>d�L=��1=DvＨ|�=9ˣ=�B=�=��=F�*=�n3�O��=z�=��	���<qT�<I�5=��=!�ѻ��5=�G�=8� >B��<u��LR(=��޽��=K�=�=an����<�>0�t�S �<���=$m�;���BR�Ռl<��H=��u �<ڼ3��N����c�]s)��ѕ=��H=��=��=�=j�����
���<�=��=d�<�3�=�->��a=�
�����=oMe=b!��=>>&�t�����w��=⫒�Ug���5k�)�M<�-�	��=N49<P��Խ��t<��<��{;Y{9=4�9=���=�>��=\��< ��=퇔=�����uY=�L�<C�=��E<T��ҷ�=�>�=�D�<fEZ<�L}=w����D=%Z*=�	����= 8ļ�K�F�+=oq��X=���<�a!=�s�<ӫe��R<�y�;�K���f=�۾��h�=I����7�]�<L�(p�:�n��C[�L4Ƚ���yW����1G�;�a�������:ű��!Q���3�;ꣽ��g��k�=�/���E��!��W���\�:�������r[��4�A��=6�=���=(�?��1,���<Q-����=��w���Y��ˡ��f	� �	����<��	�mL(��6e�2�����K�"�>��_=�xo��s��Խ��=P�>�0�{�ŽW���zTս͸ͽ��w۽(�ؽc�Y=�˼Or�=+O�<#���9D˽�xu=so�=� �;�='�c�m�=?ܝ<	灻���=d8�<�G0�#k����=Ο<�����;.=a�=F�	�'��=��%�f�?<7��<�<�=<�
���,��Ǐ;�X����%<���=��S=1bL=�@��$�
=!N����<GcW=�4=����k�=���=�H= �<��=��A���<��d���=ݼ�<o�T�C�R<zn�=�sC=��=���=�;��a`=�w
>��W�H�=�.]����=Xz�=�Q>�3=�)2��C�;�>���=vȼ7&�=PY��}��
N�=JU�+��<�~=l(=9��<sЇ���=�+=⠄<$I=J3�=B�=��=Dy=3�d��o�C2�<@����G=���=�W��0uF�5m=�	>Q]�=�Q��F��<�;p.�s}=��=�ǉ=)����4<�$��@�<�`���<(�=���"�G�A�����O���b�,��<��y=�F޽gn,=J�K�}� <ȓ=�F�;!k~�4�<�^�-<掄����=>������.��JrM=��$�6�<¬�9�<'¹=�a�<�9�;L��"f=��k<�]`�9�]=�բ�ؤ�<�]=ZV�;2v ��/{=?���%�%���>)ޓ=>Q*=J@=Z��W!�<(�ļP"W=��<L^R<�@	=r%�����)f���l=)w�<.��<�b;=Mg9=H�=�Fͺs{���J<'��=y5�=� Y=��R=א��.	>�Z8�o�2���=H�=Pj=��5=EW<�#?��4�<��K��;wM�=`K!=M��=�0;=���kƔ�-��存<���=�����WJG�SL=�!=2�;�)l=�x㻑��=k��ǵ�=>$<�;Ż���w����m���Z;=��<|<֌/��=�$�s��6�W=�H=^����=$���
�;a��<d�=�m�TR�����߽�=,�[=աf=#�\M�<P��=��<쐻��`�=kE=n�<׃�<:&��{��<��<�yU�fS��2�*��cn<�dv�Dr>�v;̹	��/�
HY=�˼8����������Q�<b���O�<�Q��-X)���X���(<R��=7c��gxI�}�}�3rx;^��=��=����(K������&�<��=[[����=P!]=�=/�>ؽ�<&QK��j=�U�=�n!:
���d<�K�=#�<o�H=��<忉<��a=�=���� ����=��ʼ @L��͎<�L
<�.��zͽq��Fc����	=җ�==������=C1C�Za�:��=�B�!��x�=�e���!<1��<��&=���=�T�� w�=�-��Ώr=�#���p��� )�O��<��=��=3Q>��;�n����@=en�<���<a�=�mD�1"R�Y�=P��=���<I�b=N�M=��<���=K��=��v����=!I��PλeO�=�F��W��,��=�P�=C����=�ݠ;��Z=E�<��M��sZ<�∼C��=^O���К��P=M��=à潀T�������/�/=[���0�콂��<X ��k/,=��������Á�U��8��93݂;�3�;�n�=�<>�
=U��<�� =�����м˔�<�5=��Ž�׽�~i=�DZ=9p�ۥ����<�G���y<o
�3��;{�+�;CA=l��/������@�;"��=�j=hn�<�T��-�<�5��ʽ� �<�ʲ<E��s�������w9���Hcb=p����":�^��;n=y��=��<1	����<p�
>ɲ����=[Ѫ��QR=`��H��=�L.>z4<di�<�
�=H1�<ˌ�<�'>lE�<eJd=��1=K��<��s=-=$�3>�қ<��<'�i��==���=}t�=U��=�˛�"�F��#x=7%>w��o�����V��
��k�=4�>��=��=ϟ<d�=��<��<���� C=�P��ؼ�|�<�eN���=�H�0ā=b�e��I����0=� 0=�A�v�Ͻ��;��<t뫼j���c�v��4�=�D����=�^V=�X��>0=�����}瞼~�8�P��-����M=Q���=@N�<$?i��m=d�2���X��S<=�R���1��k�:��=�2=�F��噆�Ʊ=�e�LT@���ý��7>��=.E�=b�w>Gc������܋=���=�V�I����w�;�Vu�����a*ҽ�`J>EH�<h�t��g�>�dP;˪���=��/�1 ýNi=^�}�����:����,�v�Z��{=�>�8.�=�> mk�<�D=%��='�<�c�=�D����>�њ=�S=1�q=�� >��>�\j>�y�={�=	�>�Rh=��G����=�����ꑼF�*�!�>3�>���<����g�u=~	��!>�\a�)W�=u�|=�3>��?>C�=MC�;�I#>�4�=�4>�����/�:��@�^���G�=o$��
w=|�*>:�=�j>��V>>-�=�v�=��2<'(ڼ-���U~����<�����4�5���ڽ���~�k�ɿ'�\�=8����Q��U<�j��An=����7=���0�X �l��������]��r����^��)?�T��՗U�(>Ce ��f=|���w�w.��=׹���(��G��vK>�N9��pj��;��a��EŽ�f�0븽��Q����0/>�ϼ�nK����j�T�D�G>��W�k��G/�'� �]��%X�`���HZ����=�#��<Lf����<��=+�.=	�
=�R�=+�H�6/��fJ���;�a��=�=6ʉ<�����:�=��{=�Ļ�8�;�ڣ=�B�Ȅ�==E�~=DhZ�'��=A}d;3�2<�}� Ë<R'~<�Bw<f�<{_�=&�7<���=���;�"�=x�]=��;�{<J�=���=�n��Oa���	;:d=@u(=��l=�� ���<��9��<n���S�=6�<��="'�<�t<�:�=!�=C�t�B����P��$�滮�<k ":��F�I74�g���Lp�腽�N#5��n��!M�JK=���k�l=�BϽ�:���a���Ͻ��	�ƛ�+�����>x��`�"W���e��'!���>��=z��=�����W-��ힽ���=����@����2��>�2I�UL�0c���z����+�<�fa�t�����">t�I��@�=��t={�A��M�=�x�����K:;����d�2Z��0�u�9t|u��ݻ='S,��!=J{�=�aѽ�il=��H��4<Ss�<2h�ڌ�;��<[��ݜ_<7f���(;��:���M�z�������ﻢ�X=�*�� z<:���O���_��<px���{�<Zφ<]��
m=��:=�Q*��Q�U��;� ��!%<G ���	+��u�<_8i=�@`��#�������޼�.\=�.&=\�<��h��U�Iļ]���'�0=X��<M耽]nS� 2N�CM�s�4���;T�=�D�9��<=@�Y<���垺�-}=HN���!�=�C�=�d���� �;S�R�;	��=٤:=1rZ�����5�<�:<�{�=q��Ϫ=�]U=���:5$ȼʷ8=��5=���=ZHּL>�'���Ǧ<}\�=P3:>��=xzۼ��<h/�=_�v���=G���f�<)��<��]�oS;�x=���T��<�^#���=����%����i}�cr	=��M������Ͻ��/��>]�=���<���=R��.du<mb:���=�j޺���뼨v��	���Y	<�����'ѽ!�#<�=2�8=13�=3�
��.}�D9�=|��;H����>�b=v�=mռ;�m�h3<�2��\Q��V�=�޾�iʸ<
ލ�ǚ>�����p��=�z�<�V,>���� �/�ok
>���O��=ZZ
>�OR=��M�Bv=�$�=��P>������=8�����߽�<=�T��W/�b=�=��=B3>el�=q��=�X~;nk�*�4�į��/$=�T�i�<�մ=��;�7eP=�rR=�`�`ӽ��o�����?�4=V
���겼�KZ�j�=И2=���=2�=2�=�ql�4�b��>W�9�ˁ���s>���<]�K=Ɔ5��_��"C���m>�{-=Ϣ0�C䅽���t��<`Mʽ#ͭ�/��=�f*��P�=0��=#>��Y�⼅��<�՟=!W�=;�j���'�kU<2���rӦ=$�S��?�=��=�f=�=��=�.<��o<s��<��S;D��$�{�R=r��<��u�M<��7�#o�<paȽQ4S���=��F�rӠ>̪c=�#>`�=�L=0K��{=+���k�;�W=�m=w�>1��xd�= +<=�=�;�ǐ=B�1=�ƽ�<3�=����D��<h�>P:�=���;����t"r>��!=�:	=�x>��ʼ;n��� =�iL���N=��	=W�<����׽����Z�9>�b�=��=c^A>Aw�=�Ku=>e/>X'�����<�v�>��:�B�>�N���X=�Fa��ݘ>M�$>�>6P�>)�>=�O>GC:?N�<�_�>��B�>S+>��<JHH>a0!>�
�>��>���>e4��?��>11>Z�>�X6>bg>@Düv3+=V��>!1�>g>qk���>8��>��>-���}��>�t�>Wn>���>��=�G>��=�N�=��i=�:8<��Ag=ˀ>7�<-��=��?�x�>e�_>cR?��,>g�/>\!�>"�A>eR��"�|=3���'𥽅{=Z�;�v���S��N����u=5"�<���<�������J�����<	{=r����-�4�<v(�<`�G� 2�<�f"��d"�\J�vv*=v�"��<¯�=�V|��eD=�&ƻ:��;(ܼ�[N����EM�=�X=�h,�3?*=�9z�yڀ�>#��=Zm=��T=�)=L�<+j�=�%`=���=c�=�Nw<��->At��?#�;P�=�����<?�=�D��b�*�@��=�ᘼP)
=7�=���ع=V��<�n��ͣ=Z뿼9?|��=w�z���e��=��G=<dX�4\� ��=ٞ�<�1"=��;�L>.>Vx=X��=�82�>���\�|�>�=є��&d=UvU����=��=4���j�0=!�==!`=<�џ= p����=�+�=� >u�><�=JzN�o�=f�=�]=҆q�׽�ar�O)�=�_Q=�r ��¼�"�=��>l�=>.M���>���=]��=FD�VRv=;��R��<����3�=j����=��=�ĕ�s'�<��\=48=D�>]��<�O��������<��=c��=��(���~<�V�<τA=Xz�<� �<{e<# �=S%J=K��=�	=���=2�����=Y(w=.��<ϖ��=T�<J��=,Z���i�=�����=�]�=C���K��=���e�=���=XoM�*���5�p��	����+=�ذ��ү�q�8;,��=�
>4ľ<H+>�!>I��<ɢ�=��Լ`s��Ԯ���=@�=�<�L=~��<V=}b�=�x|<��=��=Y0t<f��=g�P�|n�=�u>>�ܼ2�=� �=z{A����������=��=#�=�Z����=�����!�?�ļ�ý=���=��������5���=@'�;��I�ԩ$=�j<�� >E$>'���É<<�F���g�ZHG=W?�|*����<�h��"���C�=_�>�<��=�I��0�<e����'
�2�
��;i�S[�=�$t<=˙�D[3<ӼS�ڛ��b=`�<Ϳ ��+��Z;��4(>��#=t�H�Z��<sԽ�
�=[��=��=|���_�=��H�wY�<\1�=c">&+Ļ�K�=�d�=��;����b�\ﳽ�&T>@��=Cw˺g����ɼo��:6u|<��ݼ��>=���=_E��F >d(���=��+<���=�{�=z����}�o�Z��l�_=��������<��/=7V�=��Q=>�=���;�%_��,�}���L=��k=�G��#R��~=-���DK���d��'��9���
	-�~{t�vؽ4�Y=�A��|���b�7
��ץ<|l�A�\��Y콍��=3���6��ê����*<�Yy=p�	�L�#>"��߽ܵ�l{��(%�>"��`�l]��a�=����l����$*����\L����x���V��̽��8��RI=m&��]��=�ҫ=�蠼(�=/ؽ�l��f$�����&=���;!b���ݝ���:=A뽂==�p���P�N>�"�_=[�=��v=q$�Ѹ���:���'�`>����y[=o{���z��bm��"�RG���>��<�݀=�R=L��e:L>	pн[Gy�#$4�)݀�Z!����ֺ�>���=^
��f];���=��=��=ȐѾ��1>N�>��=1��>��L����E>ո7>c�g=(\�:.�=iL�=!9u��]�G�.�y�0>'�����=�H�>���=��E���=Q==}h��½q���n�m(�^�==�4�E>S�ڲ</\2����=ر��f�Z�����.�;Zj�<�U;��μ_�<'>@'�<��Q>�2m<������<-�(=Y���X=}�=M]�=,���]�<>��@�3>'�Y<F?��3]���d�;ȹ�=5��=,�:�{�P"��)ֲ<w=r�c�<x�'�Ӗ�=�)�=A	�<�gu���H��4���vn�=�C�x��hݽJ�=�� =�O7=�M�=m�.=R�}=Jw��.��S�����Hӽ�aϼ6�7����=`������:z=�����-�5�=��*���S;�	�=����{��=t�=w�=�X��߯5=F�=>��=tkc=���<��$���z=�V.=*��V�=+�>�-�l:��:#4<�R�=�g<�Y�<��8�������}���=�9@=�3�=��2��=�(}<�ښ=?I�<���<�QT�N�T=����{;!>��<7��:8��* �=|�J=7�=Fa1>�1��
1���˽%g[������<�p/�3�<9m����=go��H=��:�Ե�#�J�_�=�Ī=�Ƽc�<�pջ Q�᜛=�Y��˽hp��KI�����[����p�u�H��/V=��8�ýӯҽO��u����<��"���ۻ��5=R4=��x���<=�;:���<[� =P7i<��ֽ�i�<u#r�ڕ�<���=C;Y;��!=���Q�<�<=sl���ݼ@�i�w�=?� =�?�<#'z=�b;��L=/�=�D�<d�&=��&<��D�<�1>�l; E=���<�w4=���=�(=�[
�B'�=r�v=�7=8�=i̪=˱|=�<u=zm"�ڱ_���^=����m�>&�;��=-:C<�f�<�P<b/	=�AƼ˛X�� �[��<�r�<�zw=Y��=,�Wm>8�e=��X<gD�=@Y�<�0=��o=��8=��>lD�=���=��-�.Qt<�'�='.��yP�=��<Bj�=|�O=���=d��=�
>�
=�$��#`6=�O�;;��;��
;e�<#�ڼ��<2`=\�=7\<����+���;u��-m=fޝ�yy�<@l����<��ݺ���<j�\��]<V��o�=G��<�=[�j=��+=�*n�)3�<��=�%����k�l��<��=�H?=*������=�R�<:(��0��)�=elX��C��f�F;�V;V�<���7�ü-�$<�,�=��<�w�<r��[�)�د����d�<�JI=k]i��Ӊ;���o�t��`�s<=���}��<^��NF>tyB=�<����=m�2=�X'�D|G>Oŗ�K�<戼�cO=NЫ<Í=�"4=���=Ʀ<Ц����;��ӳ=��M=�7o�J(�=�OF=��LJѻ�<Af�=����찾=��}=01�=��+=��?=H�ԽL�=�=�=��=պy=1R��ڧ[=��̼��t=��|�>Q�յ�=G��:/a�=�5�<���=��==�K����X>W��<}C�=| �C��<�C��Đ�
q�|B�<�1�=i��<�>>�R����������uO=x�i�����;>0��/-C=d2'=D��<dp�=)0�;6>?�������<����g�=Y�i=�1>�d�<��8���,A�=h��;�5?=��=��=�~	��v=�A�=>�
=���D!=�j>��!=,��=��\=��1�ˁ��>&p=t�"���tM�=�"1=|[=Ń�<�1���J=3��<��p����=�>��<>.>�P�=R.�=2kܼ3oż����"x̼2���Pj`=.<ػ��"=��1=r���S=%܁=[�>��=�t8�hrz=�Y�=K�Ĺ�(=7^$=5F=�)��T����<�μpHY<��i<�!�����ʽ�����-��j��DG=ȗ=�Qj�.�:M�<"�-:˪�<��;t�;hh<���<������	Z�=Qw��	5���;�-;=$����>����p&<Kp�=!�s��ܛ<ԋ@��:��.�����b�
�=������{�<�������<}��ѕ�Wm�=�=��}���f=�Yu��Yz���p��l�=�7��J���kn�<=aD<�#=��?�R��o���%8,=5�m;����7M<��˼��V=J#*=B�l='���䧽�-�<��3<3�����n����]�ĉ<H����<3�c����=��<0m =5�H�����#�d�)����==��<�J=�ʹ���'�A���<�<��i=cL�=P2�7ʋ��O�<�ȡ=jf�=ֈT=�A{�\�軻�ļ�J5=�!=�(�������=c
.�uP8�X�<��9>�Fc;�!g<�Z�=)��h�:=��=���p"��/�<,6��1�X= 1 ���A<�؈=��uI�<�=";%���1z�Ɨݽ��i<�G�e��9P7�9t�Z���;�:��Ch��
�?�~�=t�ֺ��=��,=�<�ټ�@�<@+�==�켅Ĩ<AcT=�JL=j�7=e�<���#=��d=��;��96E�=���=�*�<ĩ�;��o��w����I:�9��H�����e�=Q��=�sg=҄<��;��}=&&?gF6�f��=�jQ=��"=�:k=B�n�刡: ��=5�	>��=���=tY���S5=kdx=�긼��*=EּxW�=�R��Q��>��;��E�� z��>|�=�J�=;�o���>F�I>|�X>f�>�{=�@˽n�];4)">�SŽ����&���.9=ϔͼX�< �I��>�z�>	�<��>�h1��Q�S̔>�w���䔽���� �;�h�9}�<8(���?=Ί0=��(�0��=(��=`S5=����	mi���q�kڣ�V��;��>�M�<]ǲ�|�R>Z�=v�=
�<9����t�=�q��Z��®=j��*μm$�1#t>?��=�߳��^���S�tD�=�=cfx����=�>}iz<�
">����%轝"�=��=۹=p3h���*<�o�<<6�Qk�:4_�R& �E�<����t1>��=Z[�=0_�<yp�=��ɇ<<�+=yFp=�����½-�ռPQ���l���
佅�=������ �)�t�=�>=%l��]�(>*d���﷽��z�/���&�=����\;/��t6Ͻ��,=������a�����=�gc����<�ov�;��ɏ���2�Т��"h�������a>5�ĽN�<	6��^6�E��J�;hZ����9��#�=���*�]<���=K��ܝU=�P�%��a#8����ԛϻw/_�AH:��J���׽<=�A�<Ra=Sq��#�<���l�<!�{;�%���X<��6��	ֻ��==%�A�<R���꼜�I>Kl�=n=˻ܼ�<�l=�9�;q>^=�+=�=5�=t�@=�b�<��q�׼�g�bQٻ�Rk=��6��-M<��J=���=,�[=�U�;�=T#�=PH�=*3�;����z7��Y�v=�ߔ=@�=L뷽W�}�������������B��<ي=9v�=rٌ=Jl�<Y�=H�=4�;"��mt)=�N�<�{n�_P<=Oq�=*һ�`=�B.>j�R=�~j=����<�q[=+�����<o�"=��<4�=��=�=�@��&<m�φ �q�(=0���_=;�=k�.�7x8��Hڼg�J����	n���P~<�+�;�R���M�=��u=˼��Ah%<�<�r=ȯ�=7S�=��L�F�>�	?=m֦=����=��=���<s=�b�;�����:�2T=��m��)J=�:;=��.�*�_=w�<�W��2�w�i,�<��ؼM,o=��=�p>�)k<V��=B*�Ҧ����ɼ��'>�ε��#�a�<਩<,�<�����>@���
=#��=�[k<>��=��=�?�=F)>|�=F5ܽ~��<x��*�ؿ<V�=���r�=�8�<5�T=CA�<GÝ=�������=v]=��=�>=#x�<L��{��˽�XQ=ASG�CC�;���=��m�q==%�=�=C���h�c=pe?=�����u�0������J`��A�����C<,�.�I8�=E�{����z��3�G��N<:';;��=X��<e䤼��<W������U]ý�IY���>�=N
�؋�L!��]�=�?��hN�L�����=��콚J�<��>�/�;�����=�X�<]v�p̖�ړ+=!�M=46�;*.<D�m>�UK��I�m8��&=�|ʽ�i=����(��ֽ�ӽs����P)=RKW��:��j>���<&�ǽ*�<�.���|�=�䉽�ց���	�.3,=f�w������帷/�<q��%)���E�O9=~��=�ؚ�cğ=J�A�0&�=�2�=J�=X��=O��<�氼cdx�������'=ܱ��l�=�ػ�'��ן�C����C��;��O��]���=%��Z�=��r<��K��@�<�Ћ;
�`=t�=���=�l�;�
>���=�ɶ=�2׽��y=�j���R=��J=�O�.�_�I��=���=p�">%�=q��<!�=�r�;S_J��ˉ��<ؙ����<,5=��;30>N��;uҽ�C�:��P=�ޣ<��=%O��^<����;��=T�<��=����#� >+O��v���k�=�jt=���<���<}�k=q+=:�Ľ�a;ܔ���͉>'Ն<:���*/���=�$�<�����w]�R���^Uy<��<�s�=�v"=�␼��!�4$=��=��ҽ��#=��<��R��@���4�1�g�%=���<�TY=��[>ڍ�=�ʺL�(=����pi���=q�R������w���W��[��^��<��f=뿐�/(ν4h��	\�)�o�Tv$=A3�=��i�}��=���=��<�����o>c-=">�;)��=��>=#"�=�b�<�^>��-������g�<�����>�I�<�U�=�-I<��½��<��;�)�^&{����<IY�:H+<&>�CĽ٣=��̼ 1�=��]��	�@;W)�$��.�wd���c<�gu=��=�SY=>��=M���U!�����;/��=�DM��I;c�N�7�<�G�;W�E��^ =,��}��<g��Mj[� ̱=p�<_�h;�o��o=�\��{P<�������<���:-�=v�6�P�T<�ʋ�/7,<��i��C���g=�����j��v":�1�<;�ڽk͏�$|���6���-�=�ӷ=u/=��={x_�xfj=�4��ub<��"� �N�;�i�YN=���=�ܼ�e���P�h=���o��f
��)<=�����<U�[������
9��߻��!=6P߽6�^�M�I=���=_��=�>-��;��o=R�g���,�Y;b<����y�X<��^��@�=ء=���;H�=��>m}�=K��=�>�f<p�9��D<��=]K�$֯��l�:��	�-ĥ=�������=�a��_<�y�=�|*=���eׅ=��=��-<:�
=����	=�U���g=!�>"L��g���Ȥ�� ���=g��z��8^>ed4���=�9==O�>�}&�`��=��=��h=��?�CQ���
�<�/�BG�<���<��;j���˼��h�(`ͻ=4w>@=̳Ļ��ݽ��m=�y��O�A=��i�&�v=�c�<��v<�Z=�E��q�=�����E==��Ƽ���A��<z7=��X�T��;�>�A'=P��i=��{	>g�>qu�=-C>2�W<U��<	ӽ����8�<�X��o �<	�5��g(��۽�Uc����=�C=�hY�8>bI9��ӎ;b�=�h^�tD ����S����սt���0�=�3���8P=2�]=p`J�;�h��弎�غεV=1=��6��X=��<4�<{`�=��]=R*E>�W��r	>$Ȼ��=/��=רA=�n�<�=�>�"=¼������=	q���q<�y�˻�g�=9a<��e<��/껪�=��p=��x<P�X=�/��'��=�Q�=��<: �=L��{��a��ϵ1�����������=��{��=�w=�?�w'D�,�;�E��)� �Bx�����=%`�<�C�=����Խ�<\�<B*X�u�B=�b=��=^�D>�M@�K��=�P-��f��n�=�d:=��?������{N�eߡ=�y�=��f<�D=0�=�M=���f� �@����J��,;<r����8*=;�c����=���=�.=�[�z3#=~K8=��׼���=A8��j��<�'�=�,?=��=�+�Ŕ;;�i�<�5;��m�d6{�B��=��1=�T�uw>��J��`=��8���$���=]*��5޽V�!�O��<�c⽯����&6�r�4=?�|����;2{�=ڀ���t >���� <�.��O;����6��{���;�^�>���<�R;���=Q�4�UQi=n�E��'��̺��������=�Ͻ��>�c= 4���=�̉�om�<C��x�b���=��a=�ҵ<T�`>���D��gU�=��*=���4b�`�νa[��T���<��>V�9^g<h~�<�q�=��>���<�Ž^�E=ނݽ�����G<��8=��<��'��'�<'�{=�Lg���<J\�=�-=kYû�@�!��;���c����+���x�����=�~���EQ��e�W��=�����I=|<=���=�;��;��������Oe�;��2=�+��
 Z;�p�=�X�=0���w�;=�f�F���x�q<��<}	Y<k�(�~3K=X�<^`��
�W�-�B�V=��$<H�º��=��= U׽�M�=�k��[�=;�P=�;=�(�Z-�; �U���)��s#�u��Q���#�<5�����B=�M�Y>�����!*�:ɻнe~�>;&M�k+u�~Zw����<�]�t�=eu�����>���=�.����=ڮ��l<�;���ܽ�qq=�����V%2��z�>V��=�By�Rۊ���7=��=�X߽�i���iJ>��=Q@�=��>��޽t�⽔@t=��[=+��	���s����'=/���z�\���g�>\�>�Ҧ=��>�,�=@U ���=�N��Ă=�r��z��=[2y<V!=�W�<��`�{)�#Ѥ<+F��?,2�Ǚ-���+�����A�<��<�.�A䧽ԕC�F��-��=���7*=�T�;��o=D <�^K�!�̼���<�mk=�̔9j��<#��=��<eFp�����P��i�<J��=�f���o�<�1��o�޻
��)�=�R�=�F˽$���<=%��37=O}r���e��r�=V)�<���;�T�/ڇ;;e=~e��j��(�=�Ko��,�=m����`�<�������=��.LF=�2����<�b[<��>�ލ=�Ȉ=���;\Q�=�H�=�[W�ג�=���<�*��Qq�G�=�ns�#X&>�%����E=���=�k�<���<�<�=ɴ��Lc<A��<`т<�_�(�
>~s�=� =��F��.�=�<�7�<��<�e�܍=(�i����=���=���=K���k�Fp�=M2��Ϙ���>r����ʟ=�<��U�,��='�c�,{9=�>��>�V<L){�fEB=ԏ�7�.���<Y�=p�м�~;=���'Vw�H�=�]o���!�z��=:��!�=n-ɽ�)>�P�=�9B=�<o�>���=[�=Tc�=�;N>��z=BW<�'>�����}�:=ή/��|�=14���{��[�=�{�=(��=C��=���!Xm=���<(�=ڊ�=XV6=:��=�k����=��<����^=�f�������ur����H�y^\<ń�=O��=����p<=7��J�K��̼��H=Fܟ��s�/�<ЄD�4e�=3lG:��<��<음�$N6��|�=J�=��^�68�=Ŭ�������^��=S=�am<�>�՟=��=F��<E�=ќ�=wR>��j=}�;"`&�y6k��p1���=es���,>z���(�=.*i=��=3�N�p��<�3S=��p=�R�:4<�Q�=֧�=�,�=Q��=����z㽘fټ��<�2*=:d�Q���kv=�z];��̻����U��=�Q�=��_=�d,=��=ͼ�<�zX>1z��
>�
�;�Z�=�=e�@�->5"�=�}�{伨����i��39���=v��=���!oܻI��<81�� �������Q�5�,M�=C�<��=�>�=�ԡ�OgC>2��=$�=�(��Fǀ<�l>0�ؔ	�1�I<�m_=�[�>L��k׽���<Q^�v��=�(&>�Ò�p�	�X	��=�=����:w���Ck>"��=qѻ=|�a���@#F��	������ĝ�r�޽��=���K��&ʒ�%D6m���	���;���=�w�=
 �=�ޖ=4�=���ѿ>���;1��=.����f����=N�]<)�	�[�>>Uqa>U�=6Y�=z�y��RB���*���<�9�����E=��Ͻ���>���=��y]=��=�a�<%b��d]���>ۚ><B�=d@�>�}��+�\��<�D�>�=8n���ǃ=;K�;����@��<}� ��QG>!>�<Ef�>F�C=5�=$9�=��?�К1�6��=�i���9d�+B=<�Y=��A=by�=��=H/����$=~=�d]=:��=���ۛ�]���x�=}U#>���=U�<K�=E���W�<���=�S�=Ѱ=�t=�T�=�>-=�m��w���\	��w#>B�߼4[��>�.�L>*u=wd�<� �rcn=�>����=V�!>x"�<@�>�_�<\m�= �+<����G �=��G<τ�]�:�	���}�=�>��d=�Z=|�!:A�'=xX�=�Ц=<�<�ڕ��l�åv=<p<[�=)���{�=��!=��y�b�~��JR�wPU���5>yt��e��;^��d=KO;2s�=�Y�=�`~= ~�=sμ����ЎL� ֠=�(<R5>ST�<!�X�<!,=�ɽ�ʜ=w�u���=@�1=��<��4=�k��:
���A_=��u=|��=k�->���g�"<8�2<]�S=@
��vy�����Z��;���p�;Ro���=w�^=F�ȼ:/>�@<�ʑ=e�=/E�="g��M�<�=H_<Ӈ������[��_�<R+y<��B�R,�<'Ɉ=B=�V�<!U�=��0�Xt�� �q��<+;��!=�7=�Q�;1�=�n=k�d=}h<w|&=�L��9�e=.�#=b!μ�:�1����b=�Y�=Ҵ#=x�=M�<s��;��=�פ��N=Mn�<�H=�*=,⠻ZA1<iv_=��=&=��=D�=}���sT���߆<���㓣<%9ܺ�G=n=ob����'<��=턼<Y3�=U���l��p�=��
�w}(<�,e�b��0���Ӽ�'=�]U�U���d�I�j=�#�I+p>y����J�'���G�6���������O��0Q=����7轄12�(G���J�=��=_e=iyʽ���B����p=pA=��y��i�|�.'n>�`!����^�J�G���8=Y	�$��;I������>w�%=Џ�OBt=�D����=Ld�����d��٣�F��������&���=�tr=��=���<B����)�;pXN=�\=B��=Zq�;�RF���b=*1�a�<Y-c=���;�MH=�(v���=�J�=�y��!��=��b=�����=�Z�=����Ǣ�;�n�<�3�=0�q=����)�w���3=�e1=�O��6
;5|����a=Pπ�Ep����=N֓=r\9>'��<lp���4�3�>w $=Hs�M��2:{��S�<D��L+�
�[��J�=ܳ=�9*>�i�=��x=oƙ���ѻg'h=D�<4�1=��h=%��=�%��Rm��'!�T<�!���Pf<%s�<��=0�=�e�=T�=�K�<<0�h�H=���=hzP=�(>\剽�:=��g=
�< {�w��=���Mw=�߼ǳ��g
��+J ��~��������<�
�<���=�p�<֥�:����,= ���W���E>�EH=(��<�J�<P�l=�ӈ=���,,	=Yۼ���<db��<FEI>�1[<čj���O=k��<�<���=���=p�e��j��G��aD��{����=�� =|�=��=�=�~����R��3���x��=�!�=�< �Y==YAh=��@=}�"=$'�=(��<�<*�=�ɜ����<���;ˊ>��=�.��˼!�W��=`�<@��<7)f=�-I<���=`G�=m���������=F%>��M>"�<�ݫ=k9�����=��;_<[t���<�q��r�~=���<�D�=${�=�>=�>>j�.���<�?�=U��=�w�=�3�<�_%�)�o���=�g[<P�<������5�������#�+p ;J�>41���H�`���&l���m=!x�.�W=qI=�.'=��$=��=�#=��/=���=��<�cS=&*Ľ;J�n԰�c=�+�=w���ͽ��<m:=2]E�jϋ���>K2�=�@h��2>�U��雽xh��>PRA=�8!�Uw=Q?�կ3��悼};��v=�3�=H�'}>���N��=4�%=G/�<�!&=��1<X���*�=�p_�5DO=��9=P<`;���Ê9=%�;���r�W;��=,yA<?����<��i=�<'�(�fC<yt�=��==��Ť��� �eC��o����<S盼���=��<�z5���<���=�y����+�2=b*ѽ�N�<�bK=P���}��f=���<Ok�<{�<�ʹ���nX#<�Ѽ+�����=��Ƽ�����~R;����z����:���<D����=t=�!=ClJ<���=l��<�e���K����%�����g��:�I�r2�=�g�Ա齜㦻ڔ���=b3�=W�.(G�u��=�_�=Y֢=������M<��m������1�=�L�<��=bl+=O��*�_=����S�I���>f�L=�/���GE�lm�<��8=����CBq��ڼD*ټ��#=�>EZ����<a�=M��=� �<��;� P=�=(&��<%f���k<�*�=E:���=ܯ!=햤=��<��w<N��<28
6StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_2/conv1dConv2DSStatefulPartitionedCall/mnist/fc_2/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������@�*
data_formatNCHW*
paddingVALID*
strides
2+
)StatefulPartitionedCall/mnist/fc_2/conv1d�
QStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2S
QStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
QStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose2StatefulPartitionedCall/mnist/fc_2/conv1d:output:0ZStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������@2S
QStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_2/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_2/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOpConst*
_output_shapes
:@*
dtype0*�
value�B�@*��<�����.��컚��<<;�p�!<[� =K�<�M�;7/����<��^=�j+�(�=V�$=:#�:��:5����t<��<�d<a�<*0�<�*�;��;�`�;�iV<�5亷bq=ar���7=d�<:fp<��<�/���N�<5o	<��<��Լx�<��|;�iP<�6=���<n��;������;��A<5Nm=    �[;�PO�DŒ;?v=pa*<�<f�<j+Y<�_��Z��;�e=�ٮ;G�!�2;
9StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_2/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_2/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������@2,
*StatefulPartitionedCall/mnist/fc_2/BiasAdd�
'StatefulPartitionedCall/mnist/fc_2/ReluRelu3StatefulPartitionedCall/mnist/fc_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2)
'StatefulPartitionedCall/mnist/fc_2/Relu�
+StatefulPartitionedCall/mnist/fc_3/IdentityIdentity5StatefulPartitionedCall/mnist/fc_2/Relu:activations:0*
T0*,
_output_shapes
:����������@2-
+StatefulPartitionedCall/mnist/fc_3/Identity�
1StatefulPartitionedCall/mnist/fc_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1StatefulPartitionedCall/mnist/fc_4/ExpandDims/dim�
-StatefulPartitionedCall/mnist/fc_4/ExpandDims
ExpandDims4StatefulPartitionedCall/mnist/fc_3/Identity:output:0:StatefulPartitionedCall/mnist/fc_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2/
-StatefulPartitionedCall/mnist/fc_4/ExpandDims�
PStatefulPartitionedCall/mnist/fc_4/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2R
PStatefulPartitionedCall/mnist/fc_4/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizer�
PStatefulPartitionedCall/mnist/fc_4/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose6StatefulPartitionedCall/mnist/fc_4/ExpandDims:output:0YStatefulPartitionedCall/mnist/fc_4/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:���������@�2R
PStatefulPartitionedCall/mnist/fc_4/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_4/MaxPoolMaxPoolTStatefulPartitionedCall/mnist/fc_4/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*0
_output_shapes
:���������@�*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2,
*StatefulPartitionedCall/mnist/fc_4/MaxPool�
RStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2T
RStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
RStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose3StatefulPartitionedCall/mnist/fc_4/MaxPool:output:0[StatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������@2T
RStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_4/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2,
*StatefulPartitionedCall/mnist/fc_4/Squeeze�
8StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims
ExpandDims3StatefulPartitionedCall/mnist/fc_4/Squeeze:output:0AStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@26
4StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims�
OStatefulPartitionedCall/mnist/fc_5/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2Q
OStatefulPartitionedCall/mnist/fc_5/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer�
OStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose=StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims:output:0XStatefulPartitionedCall/mnist/fc_5/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:���������@�2Q
OStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer��
6StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1Const*&
_output_shapes
:@ *
dtype0*��
value��B��@ *��u�4�j�j=|�=x���#���x=�(�xͽ���=ۚ>N�ú�D<esa;��_��4�=M�񽇍�J����5�;�'@�Ws�=���<!�?=|B���{�<6�z�[��=g�.�ƹ=�=����6���;)=��޼=�=:�Z ��q鲻N�H=|ӎ;�t����������K=\-@�)%�<6=����J��ۢ�.���f.<me@�{+�=c�$�R��:��</��=���<Be���2E�� ��&C�=a�G�c�q��,e�{�s�'��P�C�2�={��<̽��><���=��eV��&*=��ݽ^�>�����������9Pս���Z��:�%V=�!�G<�;ARV���M��B=��<N������<�:�=.4=�`л��.��={�"90�<fq�<�B~=0r�; <�=��<�ذ���q��F�L�=��	��<���K<���;��<�]�-Z�=��O=��j=&ە�	9=��<��L��b�<�v,=�w�<�x�����y�g=�x=~� WU��<
R<��" ��0X=0ks9�JM=�N)��J1����_�=h�a�)����d�`0���|/=;Rd��P�<�Ӽ(�=�;���<ڻ�.�=�J��lE�<؋<��5< �ǻ���<k؝=݁�0 9����=���=�E����漯�<�1�<'���8�:]�B<�U�;�,����=B���
;=S�<H�t��W���_�=L��<����;��<��O��7��:�=q.K<��,<�����=s��<�ז<!F������|l���Y��<
�s=����p<�
>:�m=�:3=)��r�4�檛�Vg>a	f�Р�<�bI=pa�=�f�<W�i�)5=�U<���=٪�=�V��}���;e�ʔ���;,����x������O˼^�c�S�2�J�q=�������wÿ=k��=��=|�k�);d���b�=���=�Q�<�6Ƽ�ģ<;K<���=Y7�<�G���M9w(=>�ڽ!�<_��=8^��(E;����DM½|�V��E&=�<�<e�T�Vp;�f��><.<G�|�~@�=�������<!nӽ����O���f{=��=yV���ة��@�<i�*�@��<M[M=�RO��4߽�l�<e���e��=A(<���oyX=���=YH+=oD����(==�{�$�ڼ�-�\@�=�t4=FU���9�=�~O:�=!#&�$�作�g��ț�;�<8Ck�ڳ��p��6��y%��~㌼v#�G�}��z#<f�����B�kI����<'�ƽ����ƻ���ɴ6=�i:��k�����f������=} =Z������4��z��'h�=ω׼�y�=�=Ya=ej��ի�=�l=W�=6��<㸺<r`�<�ҥ=�s��]�<"i�<�=���<B�<�{�����x����<�y=^�=i+���oE��>�������=�j������&ܐ�EDK<�4�A=¥�<���<V#�:KU��P$=M�b9����BQ��������;C����u�!=w��=I�����ƽ:��V1F=2�=r�~��p]="{�>h"k��ܻ=w�V�l����}�ᬽ{O��@U��
%�	ܽ
���6�ܙ��xɽp����f>w�+�-�>r91>vd=HQ��=�T�nN��m�>�#=l�8=�9?�1��;�a=`�m8B�Y�R�w;�,��+�P��=!�%�a��	��;T�[�-`F=�D����/;A����j�<�{�=3�Ȼw�<'�1=��=QF����=�8�=q��=B����*H�;VU���I]�
�5=�{�<�H�BL�=�U�<՘�;p���ce==�3����BD[�8� �����>�~�=sN�=]@��9<�^<Y�<1�;�y<�C>Gg�<+_\=�!�5��@>jՓ=J�b =��(�~��<�r,=�܂=R�ݽ��8���->�_=S�=r�˼H<�=��M<t�Ļս<�̷�iK=t��<e���T�N��
�Ⱥ�=F�>x��G���;��<�Y>�3�=p���8\=ݏ�<�R=Ȇ~=F�9=qǽ)��s>��<Tp��SA���2��CI���=#�l�Z $=TL"=��g��<�*q=�[�=a�	>Y�=/� <�������ċ��?׽�Z�=�U|=���=�н��ν1�黦��ʖ�<��������H�=�U0=aP`=�z�ܼQ�c�"�=h�K=�KH=�=�(A=�D���k9��*ȸ��8dl��W�=윎��	��<�OD��B`=��(=8m�= $�O璼��g��\���h=�x����<\�s��<(�ȳ�<��!=��]�#�	���^���o�RM�=pn�;�����=ڗ�=��=;8�ZG�=-|�=
�<�Α��Av��=����G�U��;n}�=��_�g�h5�n;z�Ҽg���>"�����2���=]z��I��=���h�(���<C��<�Gg�g��Ҿ>W_�<��=�����=���=^|�=`��경Å��.���K?���D�['%=e�=z&��o�%��g�=Z���#��*���I��<�U>sJ4=�J�=D�3��=V�Ž1��b�<�����IZ��\�=y�ɖ3�{>�=�@½o�=��l�~�>��`>�� �+�=
�
=�u�=�U���T���
��T���=��⽆�����<"��=g͔��G=�����9>�0mS��<.�B<af=OU�<��=gл�����S��� 4=ִ@���>}���I ��Hx=����Y�=��^=���=�ga���;
��:j[V�_�<3�d=
 ҽ2�@�ŜL>(�x<E�/=� ������MD����<>�;^��:O΀< UV=a�;�3S<�"��rEq=��:��=E�ٽ���<yd�;1Y#�}�=�	�=���=�I�9b"��ü"���/�<֦U�n�:�D'��(�=�lj<�X<�`"�s[����R�y/���r����Ѽ�cf��"=*	ؼm��<���	ٺ=ןt=�<=_��P���S��=|����=�N�<n��=(��7�=d2I�j���22�F >gA]=5��:��=�\����*�+���ʽ���x���}=���=8<<=���>�>�l*;ev/=#˽��C=H-=�Ma�]ڃ��'��8-=���<��e=k����I���t��w���(>�����ҽ�ޠ�ƨ=����G��;��w�$�޻�Ȓ����<Ԇ;�^��C==�<�Gv��W�;��<? �=�4=x����x鼁G~�4�U�!���0=�׋<������u5Ѽ�<)����V=�»���]ԇ<�k��ۙ=�]X=o�H�a
t�,�:���2���[����Ԯ�9�$�=�33�3ټ=��>������Ӽv>.�+�� P��c=U&-�x��=d��=����Eƽ����!G!��︽�g�Z=�����k���l=��<-4�=:���:\�J�}�gM%;�ab<Q��1�߽b C=ҩ�=�2=q��<9w<�A��k�x=���_�������O�?>�3�W�6;�����6|=BW���^��>�="a!<VQ=��=�3�<��;)���]��>���=�����W��=/=��=3�!�"�=nB;�����Wy=��<��Ͻ��<�{�;K�=��L��R�<����Aa�>�bs=\�{���==��<$X�=�?�=�Jo�!�[� �=T�ż�T�=��K=v,=uS་�=�x=��z=�D���h�#`��b�=2z<&�p<��ý\+�=3l>��=���=�hL����o[ѽ�%�*��<G�A�P˼<,�=�'������gÚ��"ռ($&=X�<�m���<����R倽9�r=
��=D�<�(��<�
> �<������<XT����M=�{��!�߼&MB����<o!%�J�p;}xG��L�=$�=�F��X|�<;���]{_��=9�[��R�����1;�ʡ=�F�=���=B�"���'��X���<�9ɽ����Gн��K�������<*���i�=��X=х����=�x��!]0��ֽ*��=A����%��2,>	'X��%��{3ʽտ;=��Z>�<q=��~��˼>��<Ӵw����g ]�X�%�`[u�/Ȩ<��X=lژ�.�=<"y�<R	=�JV�<�{ �O�>��?=
VH���b����=J��<�yսe~��=H��A,�7�<�Y�=�����"����=k�̼�NC=*(��8���5����ǲ<-/����׽~�*=
:ȽCrE=�};�n�;v� =' <TL�+gn�Z��={�m����=��}	�<0�&�4V'=�)λ����C����M>	G����W;�7�<f5��g����WN����n^*�Y�t=����	=�'=�d�;�
�-��<1���O=,F�<ss[=lk���V;ND�d��=�\+�U|<� �	|���Q{=�0C=���s����m�=����ð���ԼK=�����eN���@;��=���<�D�=���|�:jQ4�O�_=�
<��<q�ʽ���<
��;֜ڼ�.w=A�m�T�=y�=�x,<@ 
���D��k=���;��>8Ͻ��l�+����=Bk�=�>��@�ʦ�,�j�<bW�:A�;?v8=Q�\=U�=��=~Q�n�=ٴ���q=����׽��*=�
��k�%�M�2�$�={���Z��7��A$�=S�>K�n=�=ܽ$3�V��=N�=g�,=�j�-�t��Ӷ��T�<�=��N=����y��=��W�<Y <���=ǽ�=�>�+	=�������E�<�51��Yr=z�=ʀR=$�5�x♽<_i=��i�x�0=AG=0���*�i�=�&��C�g=urY�7Ĺ��>���-=f�u=n�=5�q���<�fۼ��Y=�U!���)<�9f;���;&���n����&=M�=w��<.�ý�gh��$ʼ�4;>��<g�������>��>��L]����3��3,>�;��\��=R� ��Ѐ���4<����J�H�z�<k9��6��v�����5�y� ����=NϪ=��t������7/=S3��ǽV�$��W(�s�����<�ֽw��B=8F>&�<e�=m�9��<�S�$�;���i��Ͻ�u��W6,�,&�<�G=B��ᓄ�zk���>ِ9�k���!&E=�z �MTռ��=� �=�.N��V6���;`��<<]û��1<�Ù�  `���>�S�<�"=�>�~OW�X�a�Ȕ���8<�����O����<�̽o�=;E�=���<57�ڪ>����U���5��/ٽk�=PC-��&q=2r�q+ݽ����y��Nk=P��\Pi�{�	�{ɯ<n��=���<U� ��?�=F`�|*��j]=r��3A9�SC=��=rnW=�5��u=	A7=�/>�L���E��2l=R�'���ν��齵�=�Vp�����A%��Z	>�)3������ڽ�8=�6f>0�=@�*=If��.yɽ�z���Ƚ`��l�-���[�&���̹lK�����T&�<l���v->��\�>��->>�����=;�:�'�=f��H�<թ�;xXo������<>6�<ʚ���~�=��4=��=����B�����V=)�O��e���<�ļV����!@�S Z��i;�3=�>Q�\Xa��-��ָ� ��=7�1<U��=��⽇�K=�����y�����1�=�+��=�5�f�!<k��=�N�
���i�ֽ	�E���\�E��9�8�<�1�BE<T?�< �w��d��1�<5^���R�����������Q[�<��>V�"='S̼d���g0˽@�ҽ{D=?�սnܧ;��0�f�=䇦=���;!�E;��E-�_#= @'<��ӼV��������=s�h=��*=D}&<�{�=��=�`��F�R�%�`=�!��K�<NZ-�I�T=ɖȽ�ǽ:e��n�j��=MoĽ �=Qˠ=�J�=�ʹ=j��=�� �9wC=F�k�<m9�=h=12T=E�=O�ནG;�=p��:�=��>R������t�2=pX���=>\>�o=��+�S���<v<O�#=�R��3R���k<%6=��=+��<٩5�ֲV<�'u�4�=�D�=K��:�#��=ݘ�=��=�V���Y��j�=��P=���� ǽFF>(�L=����i�X�ӻ>���d�&<#Lp�G$���&�=���=�O�<�Y-=����(��=X�Z=
h>'pZ�/��V4�y�\����=��z=��ƽ�O��z:�14��*�U��<'�M>NF>?U���ռJ^�=(����)��B)@<��0�G�v<Sf��x�=��<o�v=�Ò�&~����x/��a�<� <�'a��5��s˼dFʻ:��<H���&<Ũ���6��:h����ν�?F;=�z��%�/=
b=�o�D4_;���\m=�^�"�'=�H������9�8=�W�Eh�!��<*p�=ܐ#��b�;G߼4�Z;	D�;���;��;�,v=����;��=W�ż�<��=�	=D\����<����G�;� �<B�����9�d!=�S�9~*�U����Ls>��<�i�;;s=E��OC�=-�����N��ʪ<�;ս+O����e�r����Z=�t��`Zùf.�	�<�<��0<y���6�'b>�M=]k=>
)��{�=�3���q�%Q�=�s>���:��ռf��;�t���M:bni��^����� C<Z;A� \ݼ�Tg�0�<�;w�-�=xo���nK�>��=�� =�7��������`��=��ݽ��WPӺ�m�=KC�=�{��X6�=� `��y<0��=Z��[싽Y����3�(>
��=m�e=BL^;)�Z�0/ļ�e���U%�~�ż)����ü�T����!<?�(�$c�<l�>R\=�W<��O;�����p-ݼ���; ��_��<��=��= dĽ�����K=9E>m=�	H=ņ��7�G���L���MI�=��ν���;u<���S�X��<�>H�!�5>�_I�Z>�Jg�;�=/���=�*�(��=J����W���*M��N=������0�ʺA]>�H�= G=��S�^��씾��C����`����]����;I���ɿ=��:��D�;�]=���=R���㬻k�=�x��,2=-09��>�!4�A���ᠼ��J��8�<�G�<��5��_�R >�-=�c�=�4i����<�_Z�~2�=�񚻿��<�T��fH=�=���%=ɿ�<��J=���=�ֽ�&Y��&=���{�Y�|\����=s�/�Xa�W}P<bj�=�徽Km��`��R���=>��=�]�=Q��-�{��1н�$;���ս͢u����U䐽�t�<�W��wʽY">y)���>glϽ�=�>�J�������=��=�,��̋<�i	=<�<xY==�F��J=���=���<?
=K�=?픽���=1mD���޼p���âQ����=i����S��m�<ʟ�=l8=�m��D��<|B8<��'<~��<<\�_�=���=+o�<��ѽ�5z�6����0�<�(=X�g�O�3��P���=���=�y1>dO�
�o�{q�W"�<9�J<l�=R^=�#R=�e��=�-=���<���<��o=�կ=�����M[�7_=�v�g��j<��p���FO�N2�?��<���;��H=Aܽ����|��=3D�=���<��=�0�� ��m�u���E�y=�BV�D��,"��c�<��L=IX�=���=���<ٖ�=��ĩ����=�A��#>���=�k�=�����<|��#����<���=>ʨ��˽bO�<%+��Q>#��2���7���\���ٞ=R}{;.ї��=^K<R]ɼC�=�lpf<ڦZ=;l�=��E�����ʐ��#4=EQ̼��<����mU.�2�<=�M׼[D =v2,�V�=�����2����|; S�<t����~�<�)��m�%���:s�=fQ�ST�<7�C�L.=��l�n3�=�)<�<?�ؽVAX=�k�ab��M��P�i���o<)�=�ś=� %��;O�(>S�=&����P�^��>��;!��=x�=�tu�C�=�q���9;]i��r;(�=xP�=(���]YF=J���u�#=�b�=⬘=�[�#`ؽ ��=kޑ�A�	���=z	�<��Y:����Ѓ�k9=ݎV=���|��=;cӼ2)�� �Z=�ڼ��t����\����=��=���+=�s�<��ܽ�V�=�k����jki���׻��s=��ӽ4�=�Y����<X�R���>��B���3�I<t��=
�=Ȝ��Y����쿽y|�=���<퟼��A�%l����ք-��ES=
H�=D�4��"'=Ѿ�<џ�=4q�=4��<gr��½=�K��nf���="�⼿�=���<8�<;^��K�: ��<�#=󴿼���C_�=]UҼő�=ӧ9��Wûr���]�=�%�=�]ǻu�7=���;$"�4����;��^=T��;�\ʼ���?u��@4=�D�nhn�I�Žԁ�^Yt=�u=��Ǽ�Z]��B=R,�=��=B��t�B�/�>o��=�������R��<h'��!,=ï$>7`}<^M�=��Ѽ��4��ֺ=���.U>[�,>���=����z��AM�=��7�U�!=lG�!��0>=�=K�8�{�q��}-�t�n=wy�<r5<�3'=��=�+?=�;���C<RM<%�5=!"=sR<n�L������'=���-��<�l=�J����O�k�=���;��<
�<ڇN=7ۂ��5=k�o=���=�)=�=q;->t㧽X��N4<�۳���P��%��pW�=p���Ԧ���D=��=���=�x�=(�=K�ٞ�=���<��=/�8=���=[]N=�A�
U�=�*z=n$�L�Z<u��=��N=\w��S=%c�=�+_>��(��?���Z<h��=���=|lf�$n=�j�=�!N�z'�= ҟ=L4���.�<J;�<y�?=@	*=�^�=4U�=9�= o;�>	�+=�����G:�g</�=?�x���.��/���l�<�$H�F󽼪�7�u_�<B��=4��;�������_w�S�����<=YՏ��/=���<5-�=����6�;�< =��>��i�� >~_���b =T�>�S����5=	!=tN�;�6=�61=d�*<���=u(=_YA�D�=�(���H=���=%WC=�<&�=X��NF�=c�< �(�<�=��=�s�<��=oW�BM�=d{N=#��=ƫA;����<�4�<Yoͽ�>=�z<u+��ү���V�G=%4�=T6:��<Y�;�-��3g�=�YѼk����
=oT�=�Ӂ=�!�=������#�]�.�������=-�$��;��W&=�9�<^�=�B�</b0<b��&8˽��s=�9<��V=� 0��hʽ�p�=���<�k����2�^��0�=��]=�k�F��=B��=i���\�=*s*=�J���`=�ٰ����6vZ=K�n;��=��#=Ec�=�c���3��y�=���<��[��j˽��ŝ?>*��=<�:�Y�=�>x��E>��x=-��<�����Žա½�;�=�w;=@�= �vOp=�3�=�ݼ�vU�
����"�0<��>q���9�c��ϧ=M,>D���&=��_�hq�<S�g弴%2�!�м>�&�*=��=F
x�bG=8'�=D�=M��=R�<H�༾�ʼk=�Yb�}m�=���=d�-=�|;�:�!4=�aL=vuA=C��=)K�=y�ս�� �N��Ѫ=}P�=�En��7�<:�.<ח=g�n=����s��V�n=g�=g��ul�=<���^+�=v���+�w<^�=���=�F2<ȧg=S��<G�a�x:�:��$���= �=�ab�ݒؽm�:,^�=56=��_=��;ԝ$�E�T=}����q��4�S�1�me>��>�ݼ��jP��2$=�'�=��>��&=�Z�=1Gz:"�6��g��G���l<���<u�ż�9=4]��uO�����/��=W>,�5��e���7�<
g=?��=��=P�M<3������j�=YU:=�>�<4Mx=ޘ+=DӃ=�K;36=�K�=D�����=xF=�=�-{='R�=l<F�<ɝ]=o=�9�=�E�=��=�,/� ��=,!Q:�i�$�<���=��==��=�<�pk=�6G=���/��9�[<ף<�/=�Q�m��<�>q]��r%�=��5>��(=N4=xt�=
��<ұ=n]�=~��=�_v=F�r=�f�=�ҽ�zM=u�۽��>3,�=CG�=V�=.ה=�!���,=b<�=��ӽ0��<`ͽ�y���f�=)��1��G�">X��U�=�8�=�G=e<�=�=����O�=��S=��=�7W<���3��<w��;ѵQ=�vp�!0=ū�=[�<����������<@�=�a�=�X���z�9�
��=L!�<��ϼ��q����<x\ʽ�=]�=���=�]�;�ٹ=[�g�K���!�J�H���lT=�˳=��������=�|>����q4�P;�i>"i>o9�=�` >D⺽fa>U�U>�--���U��0~J>7��>�
=<*">�==��F�N�=x��=Q*A=D�<M���N�=S�S=8&��XM;I=>�7>w�Һ��}���H�w��<�2�=��=ӜŽF��=���=�e�̱�;�'��}��=�����D=���=.��w;=�E=��-=B1��HR;��=���ʒ<�>���<��>t�>��2����=����pRE<(h�<�`=$�ϻ��<)=�k�=W��=,��k�E<gXt��T�=V��=]�߼k;���~�=�=�pU=�Z�Sya=�~�=���=h�L=��H=�ф=�l�=of>ʕ>W��~� >ˍ�=ױ<E�Q��6�<�\>h�=����j"4��=�=R��=��;�������׀;��=N	��3<V��=��/���z=�=��=���Ez�>��v=��=���=��=��&=L�)���]="�'=���=̃"�)�<��=��>�1_= Q�e��<���<LR>;
��jb9�0`��T�t=K=�=��q=^����痻H �Y�R<҇�	���=�׺<"2���=p
�/I;``�=`K�=�Em�
���=���<Ce�<1����<���=�=���X�>9��=i���${O=��%��q<�+�<v!2:��9=K�=,Z��m$K<3��=�u��q��=H򼑖�<�W�=��=�j�='���ȓ=a;�<����e[��׼ ��>��=N:>="��&�X�=���<:�"�`M)����@��=�ּ=��3��nA;���=��ݽ'r�;x��<pб={R�=�@=m�.���j=�2'��e�=��>!>������X�=�)�'ot�=�=��=�~��ϼ۽�����;��=R�N���=ry�%�z=ס=�z��E�k-�=b׽��M�=�=�w>3X��eΝ;*��<��=���=U=���=�B�=]ex�na�=�K��O#��x�=XP=�{,>�1J��͜�E��=��u=4<��v�H���B!<�S�<�����H!�=j�Ľj<>g��=��<� =c�<�c���g=�/�=h�=�8>�o�<|�s=([w�Zɛ=�$�=U�d=�C	=�c�'�=U��=5���N��� =������u>ի�=�uQ=�鱼,�=4��=��?��4>����mB��᝻�ի�#mr���.=
G[=� <0YF=�}�d2e��-=r�o><es���g=�ȼ�=��[�)N�=V��T�;�7ż4󥼴�<m�;�R<%H�<��E<I�(�X��A���T���<��x�[=�ʼ����@����=�j	�+�4<��L���Ӽ"�=���n�<��W<K����A �f����st<�]�<$3c�p�=��!=�_>y�#=�e9�PO=�������=(N�<��w��4>�S�=ॊ��@5�^�[�B������!j<e��;�\�3���κ�N�=�b8>�MQ�'^a>~ݵ=y���	�J�x�3>�BR>Z�J<�y��Ƨ�1/�>!�>AN�<RG��d2j�Z	>\�>4��<��>��i�j��F=�n�%$���>=vB��B��#IP<Ȕѽ��i��j�=C>3�+�=�=�� �b!>�f���{=aO�WQz�b�%>)U =�X��^^�<ڲL=m��4d�=\Z�������v�=Xpu=���<��=P��=eMU=��G;��6��,�=���=���=j�;5�>��w;�E���"�=�ɦ�[�"=�a=��=��޼��ܽhy\�S�����=���l=��
=h�=TF�=s��=�ԑ���������K=!�>_f	=��=1},=t�0=4_�=aֻ���=���X�=�,&��}L�ˊp=��0��d���Gc������b����<�H�J�����<�H��*�<��9�J���be����<�L��/���W����K�,= ���=��_:?!�fYN=yT#=?V�<R��;~.;�Iu< r)���;��F����< �P=b>�<vM�=��<vTC�3�Z=���<W�����;s�+=�����<;Q����/�=���=@g=�#=��9=�<�ٚ=��g��Y=-�;|��=�/�=�����r��Ƽ��,<�oټj���u��x;�ݕ;���=Q^<��۽��X=%_,>-���e=���= �'���=�U����n��J�=�I5<C�=j.�=��=ĹE=h�j=&����s=�jx=��=�a(>ë�=�L˺^���o�=U�H�J��=҄�:�s[=2��<_��f���G=��=��� D�|�׼U�}="�Q=6?�=�1�Dy�<�l�`�=�>x�<]R(>p�.=�b%�0¢=S�E=���=W�<)��=�n����k<��">@K��;=���<$T�=��۽�i���6��c=��Z��	ݽar��ü4ӂ>�6��(q�<7��:��wN���,ݹ�j ��E
�㈡�T��=�'^;�Ղ��KH=��?>�R4�q�E>l#����:O>w#=c�i=o�z�\�aB�=�l{�77��~i=?
�=��=~xp>Ҥ����M1]=)���sB�=�� >g($����=��e=]��<}��;��������L`/<g�<,�=힮<8Dl�>��=���Z'
=p��=�X<�?�="��<2>pvN�+TA<�i>�6�=��n��<����Nt��Ȯ=_R�[�d<���=������	G�(=> @�==�=#�1=q��=};>cw�=l�Q��M�</�z=S�<��V>I�@=z>�<��=7�l<�>�U�=$�'��&">m�T=`�)����X����	>����g1�<l�=;e=�{^*=��=k��<V����C=L�]�H��=��C=�<�4`���==^�H=)�h��j�<Ļ�=�_=�0�xT6����=�6=ˊ!���=��|��d/>ޢ=����﮽o䥽��=��><�=R]�=0�x=vMy<&�=�Q+�f�ս��M����h�=,4��R>�*ý��}@>��=�M�w���f=H�m:֜<�sg��iȽ#� �%=�ɰ=�	���"��N�w=P�='.�<�k[<�u*���L�I���9<�N �F8�=��Y=+~�=�ȥ�d!�=�(<�VI�N��=�I�=Ke&��%Ž,>���܀�<p��=E?�=j]U��LS��TżS������=�;��ݏ��Y�����=�>�3�<wӾ������0:�K�g<��@=7V���h�;���=�2�dw=���iIA=\��=Ë.>_V�S:�tU�=E�=#kü���=��=�$Y=�>����=��=F��<l�����=;Ǽ���@G�=ŒQ��Q�;�`= �;�Mf=�T�=.U���=�%�< �����=;��=�I=��=��=�q�=�Dڽ��<M3�=!)����=���=IIo=C�:�
�<-x�=x��=�j�8�:>���������<C8x��x��袼=:=C?�=���=`�/=yA�=��='eB��<˙�=	�>'6�=��4�g��<%y���==Z�i��=�P=ڠ�=��Z����<*��=S��=<���������T��Q)�=�E���P�^oc=�H��\]=�ӊ=ir��Z�=�΅=��I�9�Q=sɑ=�ӟ=�/>�ޡ<0���qsG���1<Y�=qƶ����=A�>��G=��=0C�=�����Ͻ�F�=,�<�������<	��<�g><���~6=C>�c�<��=P����4���è���ݼ�b���Z��U4���o��u�<C3�<�Xz>��<%51���ҽ#D�/>�:RP���Գ<�f<�9��qD�<��u�j5	=�.=�=-ҏ�`kV�Zsڼ;�5��Z���h=��ǻ��8������Z=�� �lG&=�鹽犣<݆�(��v�<݃�<�sZ<��G=�=�<p=9�<&����V�<Df�<�`�6���,=V�=�<�=����6�,/>��ȡ=�\F<�X"� >������*� p���=�=5<!n�O�^�`U�=�r�=�;�v#�F�=�=l��[�=<c=��o=�(�:�	��1䓽avi�b�>��}��˽K�=��=kϷ=��u=q�Ē��d�\m�<\%�=s]ֽ�X=���=������6: ��ȼ=i��=�>�>˽Cw)�d'>\牽��; '�<��=���<'Z��bK�<�fF��I4>��f��Rב�Ԕ�=��/=�&����D���r=�$����=O�;>���=}%	>R�=!�'=�q��FY=�!==�=�ټ�u ��[�=@�[=Iv'��$(<�um�T�켖|=̎��ׇ*�̈(=���=h鮼�mR=Q
+<�o=�'S>���;��=��7>_=t+;[��Z�>�7(M�1Ѓ�Kc��ȉ�N�3=8˻:0=��=�b[>���<��=.���Sƽ\'�=j�L>۵-;���h�i=	���u7)=���=��
żǯ�sg���4=j�k�	u�=w���6��3rD=~�<*,��������p;�=��=]�:���<�rp<�^Żq!n��T����G=~A�=�b�<�>ӟ��Z��g�=�ƚ=� t�ϻ�㊽z-D��C=;xͽ�>y�=�:��]�=��(>�>�=�3�=/�=?�=��0�J�=�r#>���=;<6=�=u�h=���E�0=��<�Z=�S<(��=�!=�.��+[�=���=1=�DБ=)�;>�v=^�=����ؓ=/֊=h�)��|�=�+�=�|=��=�>Ȥ�<�>=@�X=4�)>��=���<��r=H&���=A7.>lsý�������#5>�^�=�l�<�h)=�8t�`�>�k3>Z@>2�I>#�M=x"~>vY�;��>�	��$�=���=���x���ܧ���5��>�=�zB���T��)>���=O믽����a�H=��>�lS����=�=sA�@O�=.<>�u��]��ʊ�xB�=�v=�.齆b"���@<܏���>;��=�]�<ԝ���S�=�\=w�=-*�=@9>��Y=� }=w�=�~ս�c�<���_I�<R�f=�nH=��=ˑ��/�	��	>#ŭ=�T������[�<�8=�V=aN�[�R��=0�|���=���=�<_	>dԳ=v:��;-�=^�E;��)=�Y=���=Y	�cLj��+�=o��<��&�p����F=�E�=���=������=>b<����^��=I�S���l=���<�����=+,�=�m]��(�<��}�B�=m�U=&�ȼ	���������=B��=88�=.G{��(=L�_=��=z�C��=�H�=�4=ل=H桽��=��=u�K=�C��jx�<�BX��k=��J=~{1�[����d�;����<��=��=��>�L���76=�8D=�6=*�=)a
=[;=<��.>9.E�'���&=��߻��<L-q��6=W|��׼�)D<�BT�Xq�=H=͆��wN$��w�<��2�ʨ�=l�/�ť����<�-j�B�J��<���S��9+:濄;��V<4�ں�*�����l<Ừ=ؘ�d�
��c>��ؼi9���>NW)==|};Bؤ=�<=	!<�=��<������=��p>�A���G�D���n�=�s�=W�<F曺joS���l=y�f<5��<���<r��W�%>FY1<�ܴ�o3��ܲ���=��>��q:=0<��c=e���w&�<�����{<ŵ^9yh.��C��A��0<�l;���������ҽ��I�c�=�,���N�<[Ż=�l<A��=�aG�h����O~�"� =V�P=��)>@G>k'�<���;�G�=�0k=
�J<�.>_PW<=8=}�=�y=1�ݼUIE=<̰���O=�V�=Vϼ���=�lH����=i��=A��<�J"=�;1=˜��f^�=�6����;� D��6��wݞ�8Ti�Rn�=m��:��`�g�]��>=�
>R?>�E�vt�<�[�<ӏ<�<��;=�
	=c���5�N����={������=<�Hp���!�A��<L���&���a���<m��<R�!=�Gv��ҍ9�W�=�<=fɄ�i��<A��|ф<"hO=���<�m����=����H1��6!>2��l�=��x<�S=��C=�_n=t��=�g���;>�L�U��:�T�=� ��	��=��&��ˆ=�6��d�|=�Ah�;M�X=��
=�k�=��ɽC�ͼ��%�+����=���<��1��8��
=�5����z��C�����,Q��̄<���!�3м-���V/=iZ=8h��r���2�������;�HA�/9�=��=֊<=A����L�=D��=G� ��==W�=Fr�<Q�>P�ۼ�Z��p|<��ܽ�{�=�+=��ܻg_0<be�=�N>X;=�͔�ȫ�1�<gc>~��=����!���>�u���T*�=s=	5'�)+=�'��+Cl=�=�����u.�j<�<g��=�5�<8w��r�<ᾲ�F��D)=��P<��=��'=�v�<��= <N���S����=�L���P:�� �������=�4�=��=b��=�b��Լ<��>�v�=����j��=a�%>����%�M<ȃ�=�7p=�����+a�M���>�s=�?=-<\m�=���<�>=�]<x{ҼMBk��Rj=t�彡��=�/���z�=�">�7K����==��B��=���5�`=��=�J�<L��=j�E< ,=�l�=��;wFA��?���^�?��=���=�<D/�<�j�=q��=�AX�43�=�G�ד�����=Z����p<�s`���=s�; ����=|-���/X�J8e=�
�o��:�<����$�����E�<���=�/P�?��;$���Ǐ=��H�������p��1%��܀=?�=��<=��w��=�����J+=�R�=�}�=�޽o�����J�c�<���Ѧ�=]W���=�C>߾=���=h#�V�C=w8%=��Ƽ�!��]x<9��H��;א=��;l�_=���=�K�=��};��
=��q����=��=G�O�B�B��r��p�<3e��ӎ½�->���=Z��������;R_=E=��n��腾5�$>޷�=�P5<4h>#���=�	?=K�9�9b>��6�cO�t����sO1��A<��B�wO�e��߆�%p�O�k=������=PG���@�o5�s�h޻�&�=k3������a#��
J�t�T��~Ǽ�b��<�=dg�#�;A1=\�	�@�
+s�-_��Y��G#�l`ҽB=�:'t�6 T�+9�=�0}�-<=� �<X����=�̽=�	B=��p���>�(�<��6=.�z��氼�R>���='<���������ڼ�>��y��T=?�;�`8= 6>��A�X��$c=�;�=�c���ʄڼ�3[�?�=T/V��տ<V�½��c�_���>k'��+����j=Gw�=���=V������?�M���P>�z?>Z>��8�I�K=�I/� 57�8�A�����i=���H��u����=*d��>"�\>z���6[>8L�=+I�<�%=6��=v�"�p���%
<��1�H �=��v<%"<�o>��߽�@���B������7=��=��ʻ�=�R�=�>[IT<t�=$P¼���;J��=�7<�#=W����P�������<�d"=+ư=C��=Y�N<D�(=��='v��Mܼ�=X_ <enJ���=�L�`�;i:���=�I=+X<y^=��e=�<�=���F;�N2�UI0=ő6=d����:�O���|=.i0�7L���w;���<_¢=�6<Z#~=�r<�{��P𱽳	ɼ���4.�=�~I�L���[�������x=- =��<36�0����=I��=����q���r���0��8������mʽ_$>X���5�>��<:Vϻ1=� �����=B�t�*���$��=��h=���<W�'=����M���>�=���>���<��^�g�<~��=�9=�𑺎1�=A<g��=1=�ĕ�B�z=�*=L�=yK�b0�<I!>�>�=�K���9�����=P��<V}ٽHH�_�Q>���=cㄼ!�=����!�=H��=��!�Y�e>�������c���3�ʽ6Q�n���ͽ��'�g&~�x�9�j�=� =AD<ܻ=���=�Y�=Gz=������= ��<�9��|�=܄i��C@>�=L�W�8��Ц��=5*�=>���ߎ�<�Y=/�=x�K=eڕ=��%��"˽U*=?�ʹ/���մ-=��ὝB#=�Z�*�3m�=�:%�#�S=�=���P=�(�=9�;��<���=a�=F�H<���^N��~�������=v�=_]���>}��=��<I��X��<V�m<����� 7<�:�<��2=����+�<|��� s���A<D�#<9����o׽J�0=�Y�=	�)�Cw!=�^N��&�=-��=<;��ߕѽxV��܄�=�Vļ4�=ߔ�����g�a)����;A�=;��=6ύ=����,!l�AK}��5K=�0^��r?���4������^<��"����=>_�=.�8=n��=�7.=���V�=���o��U̺�灼b�=�:�=%K�&�>V8�(XX=I4Y=������=2,]=�?>�������=,2��c�=�cν5M��~%)=�"=H�ܼ�V�=ǈ�=7k=�H�=<�=���=K�>>�3�����PC�ҊL���=YD�~��=��=+Z�=(]<=�ػA��r�*��◽`��<;Vc�@*޽M�L����=#��]�R�<t�<MI>/E��`��=	-ʼޟ���=9���+�J=O{�=־�25���j�=Edͽ���=��=\�=Y�=z�=��>]��;x-i=��&�_��=�=�\�!�|=vs��T<��'�Ï���<&�D���c<��ɼ=|�=��;r�`=��=J(��n=�3�=�����D�:ϖ=<k;ot;=mi�=���I;>ry=��=75v=T��&�"�jai�Ä)=<
<7ꜽ�"�RD�=�x+�����DM�:[O<�c=b�;BT�;c��op�:�F����ں�:�=ە�=G�I���^�\� =�	����:��{��=ͷ�=�8=�V�<|_=(��;���i��=!�<�O=��=��O��B�=�T1�ڏJ�ޝ��J*=��ͼ�3�=�s����YY�'={��;~�="#�I�4���l>��+=l7�=sO���=%&Z�(Oy�z�L�[�X�
61<vAV�����g)<T�<�=��=��v>.犽�*>H�w=�,�<h��=&����+������ ��F=�j�;(�6=[-w��UJ�
�r���=���<t��<#���3���˵���=�P�=��5�`�I=�};=�R��V�</o��c�<5��=rG��;$��O�<_�<}�ݽ�H�p��Ķt>�*�	�̽��7�L
4�h)=����K<��u<z�k=2F�=�7�=΍H��Ŗ��Q�N=��.�����q� ��u=1�Z=B[��1���2vm�	V;4>�J	�	�.>�\�=��>4X����Y����(��	#>(+
</M�5�[�u߈>z4U=���;�@>��q�<��=F����f>s�4$޽&�Ľ꒾< �6�J=B�޽ҧa�zq7��ӌ�FM��A_=5�P=�m>髳=��=m��<]�n<��<��=̃����1=��#�@=ً=�H��@�==�L���y�9$I>��|�<&e�;�� =8�]<R�)�|x�<��5��,�=��[���X=Im*�-��=2|�M$�<�<���:�X���?��J�>;�*<o�=+%�bG=kO�=���<G䭽�`
��p�;L@|��#7=p��=�2�<��<�x�=�>�/�<��=��7���:�@�=�=�.�=�?2�F{�$����W�l�E=�k=��+=ṛ=7I�=s	��%�*��|�=��!��3��${O=1c	�n��<�#�;��.����L_�<x��<?�Y=a毼��_=T��ɯ�=�UG<�$�;�K���uw<`��3��=�去XN%�Q	@��s�=� �<z�;���G|+=T����K="R���O�'-6=��=+��=��<Q���*���nd==Ѽ�L+=
�����1�7=��7�org�@('����Y-�=�� ��k�� R�Ԁ�<�@Ľڴ��
>� <xd�=��!<���<	�>ٛl�J�r�=m�==��=珔����g{�R7ѽ$`�<'��=��|=-�,=̤=�R�=�D-����=��<��e�"�i="���;J�μ�R{=�=<��`P�)>�^�=�󳼟k���d�<��>�u���=�[���=A�="ɒ<s졼|ʀ<"��:+e=S�-=8� =Y�o�]q�<>ן=�8*=����䧽b5n�v��=�j�<f��=����S�|>�����u���O=m��<��[=�:<a�=����	��|�=�ġ;��=��<�<���I>�Q��a�=�ؠ=ɑ<��!>�^=*>=��=��=Q�#=��<�J>[ߛ�~ڊ=�=�Ļ�X�=^_<��>��<l���~�)����<��z=6�*;�ϭ����7�=���=i0���K=Q	��>�a�=e�����=�.�d���o�=��N�Tu[��-�<������ڽ����(�KX4��D�=�ɨ��㦽�3�<��8<��<�#�<`�`<>�<��w;�V<�+j��W���&�=��H��(�StX=�u��-5O=(�F�"�5>c��<趝<	'o=�"�=�� ;&T��ރ�<����*����H���㫽�k�;/������4>�gQ<�5�=��8��<�M�=g�����K��н���=s��=Nj^<�ci��7&=�_f:%�ͼ23��/�=��[�]�O=Ϗ�8"��;=8�="�H��҆�｠�����B��V�=�~��6�/��=>"��=������J��7�� �C=U�<�Ԏ�[�����0>{��<�4ͽ�vh>
��>E�=JZU����=������Yؽ�O}�
e��¼`�<��M��!$��� �!���+2=����x#<AW�N���`f�=
��<���=�FN=;y<C�=�]�<|�<5�'<l&�=����H=,ȸ�ip<�4��ũ='��=C�M=N��=<R�� =i�<L�_=��=1�=�J�=�
ϼ��%=5�G��qK���=_��=���<9�D�l�F<
�m�Gf� ў=c1Q���Y=��d<[+"���½}��=ǜ�%�=�0�=Pe�<��=��=.��=�d:�+9�=h�������=m�K<n��������<N���֜�<^N����<y�=˙Ͻ"�,=��<ۆ&�Rt\�����x�=0��=�&�<��ؽd]�V��B�i��9=�Y=���=�&.<�%����O=�X��@x�<c�[<K��=��B=��=~�	�4K�=��c�Ivl�� �=��=��F<�(0�"�=x��I7�=z��=L��;���=�!H�������Խ�4D�����ſ��}7=���;G�`���=;����ͽ)e��:>Z�T=��=Ϛ.�9�����=MQü+�@�a�T=�=�Wü�.l=E׽<�>��l�W�0�=�:�=O^�<��L8FW����<�
���\�<Yr=�=� ��>�=ׇ=]������<޺����H<�׈=OQ���+�=�7��]?>oY�U���ڪ��<=O0,=�_4>>�C��-(��T�=�����<^��<��t��ɾ��P>̒>SA>*~����ڽ����~��>��<(0��7����Q����<�1��мL�Ǽy�¼}�&>�讽��>
�'=׀U<�����B��=!��C�N�W<� ��ٻW�ˇ�;V���<o<L��/�Q=JN�<*�ƻz98�~<�=H�;����W𻳟�<��̽"��:o�=c���j�<���<�n=��N*�tG�Կ�#Q <�命��<a	�=�u���I���檺��.<e�<d�"��8���<��@=H|=���<W�$�\|�<��)M���*=c�{���^<V_�=Fc='��<lLP=�_=���.O������q�<򜾼�K���U]�>�=�ֹ=���;е�=��лA��<CP)>�*=����Kܺ�����<v�>�d���+�<tNL�ʊ�=�WD9���<M:�<Ӵ�=��`=D!�<�7;7T�3�&>D9�; ��Z�<EFۼH2�<�|;��"=_K}<Yx�<Wq�;���#�^=�0=jpq��D�t>�K6�F�Y=R�M=�Z"��d><IG�=��d=�S='�=�E�	��;2�=?�V<T(�=d�ʼg@�=����{=������pq��9>� (�2
�ڰȻӜ}����=S�ʽ��'�{=߂�=�9�="��=�u(�	3Ƽ���j���N�6<�� hM=�N��Ќ�b����ռ�����0�=��,>T$����>�e=u��=_Q=�B*����դ`=c��=��>���>�=)�=���<��=�$I���>p�=0��#�:>�p�J���c>V�����14���ѽW�h�u�ɒ�0�T�&(>k���Z�&��=ݏ�=}f��0.=պv=�ԗ=ܘ���u�=kՓ��� >J>�I<'O,�������ۼ�=�=!�9�k�>���+޼s|�=y�=����;�X��Ւ� ��<}vݽe#�,�z=yWս���}F<@�=Kۤ= 5���=�W�=���<�:�<;��=(_a=\�=��q͵�hz�<:�k�no�;F�;=�Eh=�E=���<��=f��ʸg���;eM�<0c�=�V���궽�Ą�~q==��;T^���@>��%=���{B�n���(�U>{�;:$��g�c\`>)�=MxG�UY�=qԾ�� >��>zN�[>.���=#�ݽ��љ��%��齨h��[��@m�NI,��
᾵=
�o�=Վ��>`�=h�K�;�&��Ky=�9X=�i1=����I�<�@�=�

=9>�m���鼊�<�=7p�;�^:*�=J]�=aC�=;�\�}�R=���=�u��Ƽ�b��l{���c�e1v=����=��wR���y=���Ӊ���G�=�L�=��M��Ց�F�*=�;=K��<�������L<�8��ٚ<w�h<{�=!�{=E��<��>��>b	�=��&�2����=
�d��:�=�u����w=Y�p�^��c>��=�O=��ٽ���=���=�=�f��L`�Ϣ�=���+��=T٠<�m���'�٪<(X>�$�;BkE�명=@�ǽ]�@=�E;3����Ɂ��`˽��d��2���B=I��..���0�YR���o�}�s�kK�=�� ;��|��,E=$�^=��P=�	��yz���۽��<�R�w�<�c����=�6�=���<V�?>���;g�=xܕ=g�=�[�=�p�=_W�=Z���A����h���e�絹=�\:;�*;�3�<Q:���y+f��'�v��=򺇽�$	;��<���=sd"=.��<6C+�8!Y�\�ؼ�T:�F��<�)��,�����`S'=�H�<ۚ�=�����=����%���L<OPU�28
6StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_5/conv1dConv2DSStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:��������� �*
data_formatNCHW*
paddingVALID*
strides
2+
)StatefulPartitionedCall/mnist/fc_5/conv1d�
QStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2S
QStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
QStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose2StatefulPartitionedCall/mnist/fc_5/conv1d:output:0ZStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:���������� 2S
QStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_5/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_5/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpConst*
_output_shapes
: *
dtype0*�
value�B� *����-F�S�w<���:��6��\H�5���|��)����a=�,Q�	���=䕅�v"m<�Ϟ��	���Ѹ<�΅<��=-��{�j�Y��;F�����ͼ�[��h��<Hz���T����>J�<2;
9StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_5/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_5/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:���������� 2,
*StatefulPartitionedCall/mnist/fc_5/BiasAdd�
'StatefulPartitionedCall/mnist/fc_5/ReluRelu3StatefulPartitionedCall/mnist/fc_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2)
'StatefulPartitionedCall/mnist/fc_5/Relu�
8StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims
ExpandDims5StatefulPartitionedCall/mnist/fc_5/Relu:activations:0AStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 26
4StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims�
OStatefulPartitionedCall/mnist/fc_6/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2Q
OStatefulPartitionedCall/mnist/fc_6/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer�
OStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose=StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims:output:0XStatefulPartitionedCall/mnist/fc_6/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:��������� �2Q
OStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�a
6StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1Const*&
_output_shapes
:  *
dtype0*�`
value�`B�`  *�`�ֽ
դ��->��/=Qb �%w=A 2�@7��_m�<�%мN�3����P=;��<������/<'[��z%P=GS�>�CY�-��
�C�6؛��kQ=�lӽ�	�}�=	�#�-��NK�xzY��R�<��Q=��˽�+�=�=����d2>�_;|4�<ᱍ�B}�=��<�=�=�C9=�a=nC�<lC���> ������=�iM=6G�[���W�:Ge���<m����>��;��=[@0�s�;����7�=	x"=�/��>�Gl=%�ս�n�<��v�m��=Lk�<E���rܽ�ׁ���,=y�>���<�V���*>H6 ���
��Q8���i<GC�<Ŵd�`��=:W���/��/_��o�=��<v%��S�<�r8=;��؃��y=��%=���
=��=p�F:��=6���9�2;j���X��=���=�,=|�<�ܼ��w=��^��Q=�j�=6x�<�:>�阽x���3��<�|�=�u��=l�ͺ��6�m���p1�K2�p��;^P�z4���%��;F.=���M�'�b\�ϛa���F=��W�T�2��~�_�˽�X=yFi>�Z��K?��R\��p=�q>�}!;s-�����Pf=��)���7��Њ�V�i���ټ��ff�<v =8�� ��=��ʽ��O=1�8����;���=�gU�-�a=��=�1'<νѽfe���=��E�>�=�E >�K
����={0==��$<߫
�ud=đǻ&�N�&�q�@�½菜���=B�=4�<AD:�S����w�<��p�>�P��<��m�ԛ�=g���6�<��<$`��T�I�Q���Լ�
��p3=j��c(C����=Rh��C4 =�k���p<�?=�zO�w���	����PF=?5�� w4�ۇ�=�t;�楽�w�=�>�p�C<"���X;���)8�C�L��^�=���=`h=e���>᷽��v>C�j��do���->\\I=�Z.>q:���E���{=\(�=V�b,��˭E�L�{�h�A�g�.��>�	�={�Z�:�=tW�̱=l 7���=�A<��S��~= s�=za�=O�Ƚ�f<4+ٽ��=�j�=
�"=#DZ=1��G,�=zK=���F�ռl��=��V4��k(��ݿ��B����І<��[�ɻ����7��I8&=|]�����;�S^;0(����t��%�=�I�T���w������u�=ޖ0����þ�*�=�~s�<y���##��)<�&+��K�=[��:a /��AȽv���5@����=�E�=����Y�=&�4���=v��<#·�T��=QB�pO�}J�=�����=Gk�=V�R�J� >/V�;�����Q�u�j�1��<�R��\�kID=zB�=�r}�M���Ҍ��@#=�G��S��P̩=c�d�-�g��?=2T�<�=��=����X	j=�zn��t�=W-=����3Y%=�ֻ�x���弟�@<�R<��.��[Z=k{�<0W�;�G=0ƻ��=k���l�:}�Q=ufI����=gp�<���=6"��w�r�1��=�(>Uz>��M��,x=����.�ӼMJ*�ٽ�=����?�<��~=����>x�1[� &�=!��\T8>C��=5�6=f���=��u�L��:�;��<F]�J$����=���=��۽�ߌ=��p���;�|<�.��Q�bn˼��=��>*M=�
$�����V<�/->/q@��9-��]�=2Mڻ���=��������*�<'=t�ֽ4� �1�����7�g�>	e��A�=�2�Pk�в�;!5->�@����8=�ۛ��%=�Po<�Xe=��>���< ��m�K=�0��ƽ�(��Е�Az<<�?���R;><�x=*Gk��Cݽ�>:�E�$>����1���G\�K~���S�ǿۼE.�;�	��d�=��w�]��:�0P=��&���#;����2*��.
>��5�{�*�f�x�@*���B�>=C�O��<��Y��?�=Gԏ�5����;�Ƭ�0�N��=��c�W�M������f���=Dbw=�e��V��=�"��]=^4/=��4[�<��ּ5Hr=]�-��<����������6��)>�D����C=�S�=��g�0�>p�����A�:�~:��=ȝ���`��`��Ͱ*=a�`�p��Nt��>aǊ�2
v>mU�Iw=Oo��w���6�<�tƼ�o�=�@�=Q����6=�e��==�va>���;T��`rt��h=Mۅ���({I�Z	>�:�����7d=�]��#�ڽ665=�*�-��=G��=�ܽX��=�츼2��=.Ļ�^ɼ,�f�8A�	k�=�f�=�М<:	�<Uރ�����r=
��;<����{�<=r���`F@=z�9��E��Ϭ�5�ýG���S�Ž�1�=���&i��VL>%)+=���O�=�����<���< �<=ۼ7.����=Y��=��˻�Q�X�1�$�G���,>�s���<,�X����=�Ļ���qF�~�=�#�=��_����;::�=җ��ƞ��Bb�>��=�/<m����=X$�eB�<�bO:�䲼���<H"3�4��:�	h=Q0�<���)=��<��W��3.<V��=H�K=����S��;��U=Ф<�uܚ=�Ż=&4<��ݽ����,a=k�<��E�ı>��h=�Z��,�=�u���=GD�;�.��
v<N��|�<�<� ����<Q_=gD[�9�V���ν���vȁ<,̄<C?=�����Yy	�Eb �4�=��/;	U��4��=m쵼���&k�=!��=�C���=���w?�;Z4&��xl�����8��%=�^=ӏ�=�֐��X��V=�u��M<�ܴ<��̽S9�=[��x�^���<��=z���`�<�R��Z�<2�=�^�� ����\�=hdx�昙=j�<g]�������rm=�}X��C4��;����;ŏ�<h���\��{e����=�e���<�l@=V̬���.���ݼ�¼����0�TO����[��]=�^=�\����N�>�1C>����>�0=J��<�)���)=׫���=g�.=;��=n�=p�O ��}8<�����}����/=�>}�'�������P�=����� Z��&�<g$�,�罁)�	Լ=M�D=���'���['�<�j*��Q�=�\���=rd!�FG���� =���{�
><�R=ǹ�^���� �~�!=Z��=���:֯=.��=|����Ҁ=UO��c!������LX=�қ��.�;5(#�<(�<�EW=�F��i��=B~��C���A=Ӌ;C	��f>^;5�ƼQ2:T�*��B�=��w=�ҽ��<J5�IB�P��=v�-��	�}��<.4�=���=E��<����:<Fx�=sѽy����>�o�<��V��G%���F=h�=�7��ʘ=w/����=] =�"���I�Z?2�ߧP>5#�<ȅ�=���Ƶ켼t�yӶ<��"�8"��$�=uj�<:�>WBк���ʼ��Y=�d}������o����<�v�<)|=q׎= ����=O7=G=s~ƽI�E�Ȼ����ļ =U��=�h=��ڽ*Ǡ�I[ ����=�*?��ǧ�p	t=H��5�>D�̽����[J=Y�R;m[
�	|�	g���kY���.��R�����H=q�.���=��D�D<��H=�W��ю;���W���B6�=I��=��;)�����<Y�2>·���=:�ͽ�a����=���<�6L��>��|<5��u1b��ڥ�ʤ=*�@>
�>R2����0�=e���x>�]!�#��<$f��$~�= _=����#������]�����F��ؘ�2��y���RԻ�f[��<<@���Ǆ>�_3>!�;.=����=�`2<�u�=}>=��j=%ڽ�l>|��餼]�=dہ>������Q��x�	��&�;%߂=K��=%b2�I?޽���<K� �U�\>cY��n����>��<~��=f�f=����Z��P<����@���a*�=Y､��m<>����m�=ѷ�=�3����;
H�<S�=X:!=�=|��?=�Ê�/Ď�*x&�]���2c���D�����ND�=Y(����μӾB����=v)�=-��=9�M=ք3>{�a��,��2�𻁴P=>u׻L�>�b>�p�=��J=䚞=�kb�7�<Q�#�;�$<���; ��=���=��B=[5���eC��Cu=���=�h��Y+��H��>k�6k=��+�.��=k����f<��6�� ���9=:��;;D�=2G=� ]<�a����=e�f:�=�_ż�lL=╧=5���Ai�=V$=W��=�5r<H����:�يI=胒��:������뼩�|�! x=S�U�Ń-=./�=��<�t�� 'ý��=�<
�Z�C�w��=��>��뽕��<o1ƽuD�s.� ���V�ܽ�,I��=+�d=k�C=x�'�ܙ��µ���'�=u�e�;.o�g����=� i=�C >b'4�\ж9�6�=�dN�w��;
J��ژ>�6�<����3,m�-C�=���<���=�.�L��<,�5�*������<Jݼ�k=��i�
��=�����K=�[#����U隽��c5��)o��>0���=�0�<�2�=«�;�U�=��Ӻѩ����=W�t=�=+�ؽd��A����m���>�b<��<��=��=u���.λ���J&`='|{�����z�?����+�F�^<�gw��ͻ\���=g 5<E=��k=��+=N[�<��o���̼wo�:V�5��=�l�i9<7~.<*+a�݉=#iI�MЋ=6�νf��G4U<x����R=g�T=U��<-�W��:�2��ÅA�(�=/�����$����<x���i����X�<?:�� ��<s����>�� =��}�2ɽ�*S=dYT�4Y�<{��<�C�<�����=�iL>RS༧<N=�`�;�W��SD==����7=�{꽝���i&�=�sV���s<؂���,=3A:�'���='F=�Ħ��8���A�<vN���bw��J��#G�=�W�tu}=A3����;�h�;~�</�=US�9�t�=��;�@=��l�<�>�=*U>nX��V �#�˽��0�kK��1\H�ß׽���=�i[=��X��,��������a�o>4V�=�m#���8���>����q�5>͉�Ѽm��5m>j��<��]�{�.��)ɽ�W< َ=����bٺ���m�<���<�7��j��=�r�=�*�<�-���7����=#�����P�1�����=i��=i��zG�=b�=!���av�=C�[=�k�=8Y��	n�=n��`%�;�׼_���c�z<��=��ټ����5���3�<=G�<���Ï$�!+e=4�==�Ę=�|��D�	>,�=�7��(=Ng�==̎=JJd=�B�>�%=1�Q�m&^�v�l=}�Խ��ڼ�g�3]�=�Y��F���U���C�@n<�i<��=���< 6ݼj��'=45��wgټ�m�=�q=��K�^<�ɽD�;b��=�U��
�X��6���?ʻ\�ӽ��V=�2��f���MMֽa�<�����+<�8UGȽ�a�8G󽬭4=Ѐ;>�t=����z��;�A<���=����ѽ����宼���3K����4<�Q\��+нﶣ=Bf�=&u@<1��Hή<f�>^]�1 ��QF�kμ�<�2�G�>=̿�<�fL=uߝ=� �=��ս��s����92 =����D�=��<+�<�7F=��+<9��aM�f=��A��8���z<�%:�Qg���=�w���Ʉ;�ߑ<0�A�!��6~�U=�i�=�*��v�����<��=]H�M#,�s�=������=�<�ӽ�sE=C&���������_�=�@vZ�f� =�0�g>�	=�qQ=�=<܀�<8ӡ�j��=�}�=o��r����_�<<�@��20=��:�����=h\�>Ԫ=FI[��F�<୤���<��2�1��=t	ʼ��=Ⓓ�D��몡�-CǼ>0 �׊^=�Z�=<E=f��;ɝ*��t�<�Y=�j�����<�;�(� �~�GQ=[u��<��=U��LZǽ�ǈ��+N<�W� �'=[U���=cr���Q�=��=�͡��6@��`�����<3I=3��=Bw ��`�=t�=Fa=)�<��[��ỽ��h=vo=u�׺� C�"��(�}/�=xh����>�׾�9�O{�=�0=�~c�sN��=&Sm=ӏ�=PT<s�=��U=����Π=�d3��n���d�=%w�=5��;;L�<F��=+��=U����<�S-�o)��Qx�=���<x�=]�<�a>�F	=����'���1=W7����<Z<��pt�=n�	�"=�=����&�;��ý��<F��;�B�<��;4c=⽘�&=��u�==s8��.=S�=�)��=�$�<'o��/��;�S�=͉�+~��<�!�=zo���0�<�����`�=߱#���6=���=>R�<�o;g��#(�=��=�@c�lL�<��=ϩ�� .�=%P>�=j=T��<-j>a%�<�Η<�P�=���=$�1=XB�{Z�=���������R�>Z񗽬cM=fF���l=���λ=�,=�U�=�����,�:�}=�U�����%�s=Č�<2���ͻ�=U�b�q5������ �I=�t>�fƽ�`�="=\�=�,^�w����8�<��ʺGѼ�θ=XȽs�ϼ�+B�V�<��+]�=mM�:Z�&=q^�<�ѽ���=YxӼ�_۽�%j=k�=_s�;�O�=�VŽ�6��B�;_܆<9ż����Gv�	��=��X�@�T<��h�$�=xyV��`e=����=��.ܩ�J_�=a�~�c�=Mc~=:�?=�~��j=����Ľ�	��� ����=�����<;����C׼d��<�c=���<��;���7=��=�o,����=Hav�tZ�;�Ke�M4�=i��D�c=L��=̘ =�jS�+ߠ��[�=��!�=1.�=d���/�A�r���=��,>������8��<�:�<��m�z�W�f���u���N��=�u���=�����=v��;{�?��ݠ�Wa=lOH=/�ɻz,����C<=i=���=.q=K��<�l�;����o�#=�+��@��j��(l�=nɼ)��=W�<=��Ȼ �<�>�`�;�NG���&>}�<r =�<��=�E�;����܆�`��Ԗ���t&:��1�P@l=�ˑ�i�=��=�៻NQ�<�񽻆0;�^��$��'ю=��>���gi<�D���;켽s����O;�5��
ȼ�̃=�ʮ=����&�;��<�~��>λ�J��9=/z6�@���Zg���=WPm�"A�=���<����Eռ��$�Mu��^b���佽r�=�h=��d��6�<}��2eW=6}��n��.���Q�»�6�=��]A<A�!�N=7�S<��<�3����+=k�˽�ZP<w���=����ͽ_��= q�<�g�R��a߽�M=D$�,�Ž�E�<x��=g����	�=�R\����7�=|f�s*��Ğ<�.�}|P��E�< ����ǆ;�2�=h��Z�>fa��n�eK�<�|�<}����=3�ּp�e�>�/�φ�_O弓�k=i��<�����h<u�=:�ڹ�}/=EHY=W���w��=�H�=�t<%=p���r��=8Y�I��=� z�kN�/?�=@C=����М�uv<��=�U�<u(¼�H�=uʴ=�����~{<i)>�`��������:C�9=+4���=ыk����;�h(�Ľ,zH��TٻQ狼�r��u�n=���<�+Z�������ʻ�
�=���<���=q᜼C��=����:=�.��+���,>1y���3ϻ(��CH�/1"=B���1����#�q�=��������~�\	��I2`=�tB�w�׼���ޤ���1�>�u=hn={_����&����������.��nT����=Zuw<��x��~=�L�c�=�`>ų��M���y�=�bh�t��=N>��Tc=���=ִ�=�Ҽ��s����;ܨ�=�wὯ^�=;桺V��!�#�R9�=�>Ww=�W,=
�=�K5>v��<V�ǽh�>�X��",>I	��F�m,�=��>I��=Ml���<}L�<����<ZL�<5g~��8����=O��=�=�8;�2c=��꽻�Z��Ȑ����<tx�=�h����1�H��=5�=/���
+=3pi��=^��L������|�=���;vo�O�=�P�=?(���=�m_=�&�c�&��7u=��༩��=��>]=���<K=�v�=������ʩ= �<l�f�6�q��=���<{�;�z������M�犑<�~K=�<A��G�u;7��=t�<8�~=�����>_�?�yE�=3/��	<)~s=�<T>�=��T;��=�s�S��&������=8����f<��"w=�5�<�y�=U��lc=���V�ɽk:�=!�=�V�<S�=���<�&>����?��=�y��P�</�<&��=Z�����j����u=�="O����=Y�J���<TSa���?>,6B����=ӱ=<��j>"�P�����=�$�+4�=�
>Z/�=��=4}N��0Ƚ��>>s/�����=��нA�,�y��=uX�=�C<T�F��)P����������=�b+���s��L.�j�u=�j=�iI<��F=N�<�H�=Y�����5�8<�b�Ғ=V�Q=��+��B?=h�>=/d�ʦ�P8���=ߛp�u�?����=�S�[�$<M0㼜❼�L�ݢ��p��=ePm�p�;�3���y��Gƽ��ǽB�96��=iY<4�}<1j�;Z;�
@��޽"�=*ש=��<I�=}h=�n�����=h�f=�����7<ѢW=�ڝ;�3T�������=��=���<�eQ��D��/i=���<`F�� �=����=�U�=����	���Ai�����,4��=�;=�~�<��ܽe�=U�=���=,�c=��>G�>P�_��9�#>���<RE<=�����C�=����̄=��~v���ܼ���<5��<���C�P�?{=�a��kT��^�<�O�=y}=�)ȼ��Q=%�M> ��=���=�2�TN>C���]8<J��ZݼD��=����#=k��ŦB������h��u�<a
�<�a>/=go��瞈>Na��x>CC�=��Q>m�-���O��:t�h��=m.=�`潕��=��=�p=E&н4�R=�����D=�!�<���++�=��=����M��-~=�,�=��U�}�%>Q���qӽ3����u�;��>=��<��;�k=ڬb>X諾I��;�71=�L��}0�=��udX<f����U��g;2ԭ��0��*�R���o��<�&�"&��B<%9i;!��=��3�UTS<��=���<"�j=֦��������݁]�^8 �r�Ǽ���fA.���2_ۻ���;t�x�!<�*�=�'�si�<�z�=��7�3T>���`������K!=�O���S�̷�=��L��G��髶�q4���7=M#(��F:=k_�=7ヾ�R=����L �JLP�9O>V�开b�=l�W�A��u���C<u��<y:׼}ɂ=B��=���[	<E'<=2�*����<��=��R��d<+�e=FdB=���[�>AR7=(�߽�֏=�VJ��{��!%ʽw�=�T?��<~=���<Q^=9膽p>��A�=�d�;rn+�VD��s�;_蒽b��=D1��o`��a�;;b�=1Y=�=7�r=[;�="��+�½ۼ==�*�hw�=���N��={r1�� ��;��ȼ�M��>}#ڼ�Z8������L���9�g >&�>���L��<B��<m�Ͻ���= �� *=>b�='��<'�ӽ�۽S��<j)<�-�=n3Լ�pL���(;f�=��>Kk޼{�>#��=(�d>R�七� ��=e&��N�+=̽2�=��a<��6��⦼�>q�@�A	=i4�~�F=M_�=�$�=��;w<=6����=&�ü���<Pꟼa�@ѩ��l�<IWR=y[8=���/�=��=��:�5 =w.w<��i�)Z�=�ac=Ҁ�>y� >�DýR :{U�>��i����=�=ȽG�=! <��=`i�=e�������<@ c��b�=F�A=���ӕ�<:z�;�K�=� �<�>��6>��i>Y=3��<@[�=�㰽�+�=IJ�.&0=)�=S��<"�%=H�=x�=��=�`�=g���C�����<3n����>=ǛO=�x�=d����y�=�3������	Q��C�<@��<�i^�U�F���=��=<3Q��+Ļ�>�����=�=�Pr=8��=8�=���=P�N=p$}�ͳ�<�H=��<�W�=<fO=�#��ؼ<���;��8�\H�6�=m�	��N~�C�<�R�~=ߚĽ[#��y6>u�%��]!�B4�,�<hԽ��=�+���=T��=�>]��<d�<�=�<ڻA=���<�yɼ6C�;p�:vy�<<`�;s��=���=����x�=�I�<�F��X���[;���=C�������<E����=l=�*�����=W�}��ۘ;�)=���=�bx�
e=h��=&@G="�B�f\�<�s>�=�`�v�=޵½�۫=�l�$�4<,ռ�Oν� ���ϝ=��=�t�=��+�t�=p��<�̻����B�>	=��qC�<�{�=�D�=$��<�%N=;q=�h�<ye�<k��=(k=���U<�=�Z�9{Y�����=��<)��7��xյ;i��}���_1<�Y3=�� =\��:	���N�=P=�m=��=_OD������l�=ff*>eb�<�aW:@_�<^��<�`<r��=��q<ίO=����SS,�I�^��s;oy�<��X<G�-=�Q�<;��"7w���r��s�t�����+<$�=�OS����<�ۼ57E�v�=m�=���=<�&=G3��6��=���=.�u�A�<���=��7>&�<�ˍ��:<�]=
ȋ=����K~��/�<�dr=ڽ�=���<�ʐ��?\�:`=	Ψ=(�޼d.8;�TM��<|м��;~$>6ށ�W��=��$=���Y�d<�=��>=5��P�=���<%Y�=-ͼ�_�<�=�O�=[�Ž�,��-��7�:>�%n�sU=gI�����i��Tܼ併=,!L<9�n��=����<+��4�K=�M�<����z�<���<���=/p�=UQo=���<��<�nf=Z�>������N������=I�~�|��;'~=���={Qs=
�>H�������񽂏�<����!����I��G=ֹ�;o�U=�~-�q��=m�{<\�~�`>`` �������=0b=y�=�D�=A�=@��<��b~�=�zc=�,�;,�<�:>.�<�5o���X��c�����֯�����3����>�e/��7�=��׻;:��	>+>ؤ=������=�/>�g�<����=�;�=��>%��=��漟R2�p]�=�v)>�d<���=et����=lo����>1
��t�����)�(��=�ٽ�(���v�1��=�E��iy>u�=�0�=s��<2/L�n=i�a=��P�ϵֺ򣶽�W�=��h���=�p��Ͻ\��=�AK=aƌ��Wؽ�/�4)�=������=�D���Y/�	�v=g���M�	>�F���mr=��;�S>�(��:�w�M��T��ڲ=���<Dy��	a��q%>�"|:��O�*C=C����(q�����a*н �=�Y<.7=�|>�~=-�G&�Э��q�>($#<4j���7����L�=�\�u�5�b������=���>^g;^�����=��5=�1����=u�����=�G*�{t=�M�����=���<�̽X=j�<��=xϴ��"=��;��ڽ3�|���="Ľ	�\�۽2����<��=���<�J<#Y =�r���<28
6StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_6/conv1dConv2DSStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:��������� �*
data_formatNCHW*
paddingVALID*
strides
2+
)StatefulPartitionedCall/mnist/fc_6/conv1d�
QStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2S
QStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
QStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose2StatefulPartitionedCall/mnist/fc_6/conv1d:output:0ZStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:���������� 2S
QStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_6/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_6/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpConst*
_output_shapes
: *
dtype0*�
value�B� *��m�=��Ϭ=�W�;�����ջ��<<�u;�����U��i�����i�<r#=kK��q�R���۰0���-=�����I��V�<ml�ļ=��="�<�䈼�bټs������=���2;
9StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_6/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_6/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:���������� 2,
*StatefulPartitionedCall/mnist/fc_6/BiasAdd�
'StatefulPartitionedCall/mnist/fc_6/ReluRelu3StatefulPartitionedCall/mnist/fc_6/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2)
'StatefulPartitionedCall/mnist/fc_6/Relu�
+StatefulPartitionedCall/mnist/fc_7/IdentityIdentity5StatefulPartitionedCall/mnist/fc_6/Relu:activations:0*
T0*,
_output_shapes
:���������� 2-
+StatefulPartitionedCall/mnist/fc_7/Identity�
1StatefulPartitionedCall/mnist/fc_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1StatefulPartitionedCall/mnist/fc_8/ExpandDims/dim�
-StatefulPartitionedCall/mnist/fc_8/ExpandDims
ExpandDims4StatefulPartitionedCall/mnist/fc_7/Identity:output:0:StatefulPartitionedCall/mnist/fc_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2/
-StatefulPartitionedCall/mnist/fc_8/ExpandDims�
PStatefulPartitionedCall/mnist/fc_8/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2R
PStatefulPartitionedCall/mnist/fc_8/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizer�
PStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose6StatefulPartitionedCall/mnist/fc_8/ExpandDims:output:0YStatefulPartitionedCall/mnist/fc_8/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:��������� �2R
PStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_8/MaxPoolMaxPoolTStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*0
_output_shapes
:��������� �*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2,
*StatefulPartitionedCall/mnist/fc_8/MaxPool�
RStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2T
RStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
RStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose3StatefulPartitionedCall/mnist/fc_8/MaxPool:output:0[StatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:���������� 2T
RStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_8/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2,
*StatefulPartitionedCall/mnist/fc_8/Squeeze�
8StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims
ExpandDims3StatefulPartitionedCall/mnist/fc_8/Squeeze:output:0AStatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 26
4StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims�
OStatefulPartitionedCall/mnist/fc_9/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2Q
OStatefulPartitionedCall/mnist/fc_9/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer�
OStatefulPartitionedCall/mnist/fc_9/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose=StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims:output:0XStatefulPartitionedCall/mnist/fc_9/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:��������� �2Q
OStatefulPartitionedCall/mnist/fc_9/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�1
6StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1Const*&
_output_shapes
: *
dtype0*�0
value�0B�0 *�0��\�|������q�֜�<z��;�a���:p'
�P�B=۰���� >O��L��L�T-<J�,�d�ؼ�Hҽ�.<��=I	��Zr	>Y7L=�{?�xH;>R����ż�D�P��- �z#�=�g�=c���)|<J�<�t�;�F��7�=S>=c߇�<&>��纺>na��ꋛ=2��=wS<p��,Z�i*�=��<�BϽ��Q<��=�|>����=��=S��Ⴝ��Z��*>���=�_=E��ڊ���㢽NK���j�Ⱪ=�ʏ=�{�<y1<>�&>��軄RX=hd ��Oս��=��Z�0��A�g�l��=T惽Q����[>��3=�G����_>������g>2<�5��t�=�y|=��=�w ��A!��н�*�C�=�{���Y<�k����<&<:������ٽ���b������	�q��{�<(��=%eF���%�����=ӫ=E��<46���/ӽ.w!<ϫ��~+��8}=ߘ�2���M>��Q�*=��=�*=�a���s��/Ҽв=I��2�Ƽ�yb=��5��U�<;.u��a4��x*�nY�=�Y�=g��<J�<�Ϻ�
	S=_T��݊#<p����cb<
�3�a�;����t�=�!T�Gj9�O�)=c<=|;�<�c���M;+_��J�<������=DƏ��f��*Z=���=��<'WT=@R廁2�<�?���UC��#�E&��X��Pl)=�==K����\��fE�oѽ��>=��o�r�佪�Q=ͭ�;wY �=��DW�<��S>�|���=�T�=Q���fk�=>��=�y�=r3�<�-��t}��_���,C�b=�԰4�>��=��=�N�5*>&6�=#�<�p�I�=t�=]/#=�I=��ǽ�޺=aŊ<��>=���<$�	�=�E6<����N�==RA=	�Լ?1��X���ܼ���<�5�<�>!�K̼Z)t;ɀ��"
<�*�QVƼ�خ�mBz���=�D�= �=BPS�Ȥ�{�	�K����<�^�=�8=c��lE��`��=�͐��;��d�n�(��-F6�K�����1��o�
�0؆�KK����:=9�a9�<��=�$�;g��=��<�x`;٪+<�\S�H�(;CG��rUe�rr�=��o��́�s]� JR�T�⽖�>��=,�Z1a>U�~��[:�rS*<.�2>��3>
��#�@<�}�=�8нT\i����]����#=��j��rn<��u��V$��1�<��=/�+���g��Fڼ[�n������=�"�=�>�<O���\�'>�����f�=f.=�+Ȼ�:�z�=���=�=��w�R���ؽ5!R��%��}<��=��=3�=��k=s\6������uD�w�W��$�ϻ=k�#=r�=�pj=)]�=�Y��K�<�v�=�==vf=O �=�n�:�=Ғ=�`��/l��s.<%�C<�ܽ�-=[�ܽٯ�!�_#>8��=Q#]���\>TP��z�U=S����-P�D1)>�3<m�А��@�=:��<��j���,����'!�=�)����q�N3��δ � ⱹsu��6�׽�mG�\�����lQ�?4�#�_�@�)����=��0>G'�^IA>,�n�0��S����o�^L==�P��"�U�y�X�t<���F�<��l��� =~f]���=���=ag���E=�B�8F�=�m	>�K�Z�L���e�۠�=�@	�����W��<G>7���<�%���=�϶��U-=k㪽�e�=^>xC��<O�2��5�ս��>��<�ʹ<��T=P
u��H>(:��l<R�Խ]맽ˢ�=�q�=$��<�X�<��<���;@;��l��2��w��h�#��9`C��l�ѻF0�/Ż�@�=x�2���(򕽳�~�ƽ�(b�1ʞ=�P�;�2�=�G�(��=n��}�c=��}��,Ƚ��Խ���=���/aݽ�>�좺�J��]BD=��<���=5[�#���9�<M� ��l��㌽)�y=gi	��U�Q4�螯��/����;��.��@<�DY=%�������4w���3����=�Q�=]=�E�=���<� �yG�K���GR�yc=#��M�"<G�!�{zT=�@=�s���P4=,��<�M3=q�6�)�Փ|=rқ=��W�0��=F50����=V�Z�'�]� ȇ<����x<����dҺ���=��J<춻���^<���<�Ѯ;M��<��N��h�<�����h<i�=S<.��11�	�ʽ��<rʻ���Q�n��Z<<mZ�pI$����=�����,=����]<J.
�qI=A�P=�p;k���m����r���'=1%�=A;뼩1�=�t=�=|%�9ٳ���t:y��=i�=8��<����9��Q��=C�[9��&�s����7<�_�J��=��e;�=={�=���<�k��V��:U�=���=�}��;������K=�{�=���<���=9ҽi�� ��_�����=�"=�Tc��u�Я��Tn����<��#=oǽ���<&vI<d[=��?��r���<,t:{W�=qg=tJ���C��*W��/��<�y�=R��<��;Ix��أg�h���Dޠ=wg����<���kp�;��<#2��K>�,t=�]:I=6':���=���=-�����e�&�}�׻3�B�j�,=B9�=IT�=����k�=�#\=���l�"��e��l�M�=���;���W�<�	=��;��=x�	�	n�<�"���<'���H=������˽*D>�=�u��7�=��a�G>b�<J?��T��m<e��;��,��F�񷋽)؟���S9�=��Ἄ0���*<z��=�;S��%|�r@�6z��]7���=8
�=��=!'�ie>Q�=�\^<��~�Fʭ�.Y=|��<�����<>{��=�=ɑ��{�=��W=Qz�*03<�y<�r�<��=�����m��f�f�-��<o������� �=�p^=�Y�̨�o�K;W��<���;��=/��=��н�#Z=��:=��6=�r|;E�����=·�=5j�S�I=TE==�{�+�:L�Ѽ`�=��<����*�O��-"��'���,�;����᲻d��h*���tE=�3��QP�y�����S׽��,�̼>kĽ��=M��=����<�
��D���B��\��,�����<����;���!+*���o<��=Տ~=�֬=ĞZ������ <�����;��=�L��估S�;�,��׼�x�=b[	������B�=�B�; �<�l;�˴<�i =?x�=V�2<���<�,H�,B���>zn���a&�5 潴���@
��8=�w�<:��<���=B����	��@R�N�Q>H>d+$=R��,��<��<ط%=-a=m�<v��̓=0%=�F=F@;c<�<�s=��u=��&=�?=|��<T�=燣����\�*��� �ә|�{p�=r����= 5+�1����d%	��[���E�<�=c��=�<zM��(H��k���"ν]�<����=������=�P�=�;��/;=[�<�K�ϴ�<e�t=��d���p~�i������p�<��=U3���=Gϖ���R=|��=T7�Bz'���=N��.,v=ɡ���<�o=��`<�%�<z��<��ν�M�<�.D=4�#</d%;sA �:�<�jj<�挼�0�=TY���0��Q�=�����g�<M���e��;��s=!�>UB����t���Nd�=Xʢ��XĻ|z|�y�����<�� ��Ҵ<	%i���⪼N�0��=A3�d'�󉦻�����u�=��f���=�L���Ƽl�4<Iv�=��13�<��9<�G������T�=��=F �� q=a=Η'�F~ĽF8 >�o��Aź��=/�.�=3���|n���<�Ϊ�!ta��
���U�2AE�韇=��<��ǽzR>��;�/;��v�=F�K�@=�R�=�6�=�Ҽ�����;���I���<߻�[�=e!<��߻1+�>B��܅Z<���ig�=�':��g1��A��`��=�Px�2��=Wq2=`��=磕����==]���8�=�9�=�J�襢=��Y>[˛=QR��.<�<�g�>�8Z�����C�����j�=GG�Z`s=����2j�Qs�=c��쉽ڐ�=��=�=B�5��؁�V�����	�q�N=���=Kj]��+�=jk�=�Vh�s��=`��>2�-�Pʚ�9�!���=F
���P>z@>uI�pY>x38<jP>=�<���{�=��=$g�=c��q�ռ�<E>fǱ=]�->�y�=/��=�
�=��$=��5<��=��<3��-��=�
\����=7?�=+[���A=|��=2+Ƚ���=Q�ǽ�I=��4b�=�>���<�>!>S����@<��wr��>�m=u�=�q=�,���~��d@�=����.r���&U=���w�=�I�<jh�<�Y�=�l���a=��=�qϼ$)=8VQ=^�.��*<7e~����3,X<��<�ʹ;�1P�U �j8��Tr�z�{=�� =�+��`����$=��P��6>>��=I �=���=d�<{=}��=]����ῼ\J%�!+S<��=��<��<=Ge<w5>����� �=�=Yr��CM=-�)>l׽<۰#=%f�'�=�U>��=_�ֺ��)=XS%=�:�<.`=r��=��=�,�ּ�Q8�od=�L���u��04=P5<"�N<���=��(�Lw{<�N =,]��u+�=܆� �ۼ:@��(��P�]=.dt=��=��!=y,�=�)<��\�Us���ݼ�=��%����|m�<�&=\\=sS ����<��=U�����������������<���=��j=����>��K��p�t�t>=)�D���=����?a<*��=�ר=\��9�=rn�=."�������<" ��7��=���*d��@=�.�;\��=S�:���<5��0��w*<�Z2�_��=�Y��O>�G=���<e�?����=����-<������Ͻf�=�UM>��`=t_��U��FC�TkP=� �@��<�����8�f��'�=�2ڽ �ټ�[�=^ ݼl�[����*-:>�7��R�>���<d6�	�>�I���s>2ż��V>¶Ļ��<܌>9b�=c#F=�D=}:k��㴽��7=�Qѽ|C=N��<O�j�?��<�?�=%�༱Y�<Aq�=��<���=���8z�<WE���=���=�4��7��=ɕ=�$�}}L=5A�=�6=��g<� =޼d=�4��1�=� ���L�=���;�=$D��ڬ�=���=�������=E2>E���M���g����L>�8�;n">g䵽r�4=��^=:���"���}� �<�F�;�4l<�ח��v��J�����=;�<�Tw<hz��=A�>z���և2=�Zt��W�_>�h��G�>Th=2��= Y�;�k����>,��4Ӧ���,�^�=�k�e��=�=��e>�*=nd��]�=���<�47=���.���'H�=D�=/�N<��=X�>Dz�-�=�=�7�=��=n
��O�>���=�;�y��F�3���=���=7M�=a�J<���<�^s=��
>=⧼��=wZ�=���m�=�-�=��<=]=%��;�B>R�<!:�=f�;=6���<�Ge=��-���9�/�<�M�f��=�oջ�>RS�=wL��FS��ҙ=��=�,=���=��l���9:7���QE�=>�D#�f4=�T�=�W���p��I�<�5O>x��=�&>|YC<�dI�L#
�w���\)��RR=|A߼|�~�BT�=����#C=�A��DŽ(~GG����"Iｻe:>=jy<��=Ć^�ڀ�=Ec>����Mˢ=��=�!>R/��8��<��>\S�VG�=�̻��߼�g�<&��;b�򽂳�=��;=����>�W.h=�� = Cv<��;Cը�M ,=`1>28
6StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_9/conv1dConv2DSStatefulPartitionedCall/mnist/fc_9/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
data_formatNCHW*
paddingVALID*
strides
2+
)StatefulPartitionedCall/mnist/fc_9/conv1d�
QStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2S
QStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
QStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose2StatefulPartitionedCall/mnist/fc_9/conv1d:output:0ZStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������2S
QStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_9/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_9/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOpConst*
_output_shapes
:*
dtype0*U
valueLBJ*@)B��^�=��E�6�T�(���N��=SA��j���{zT����<�齽�[���ߍ���V=X@��88ۼ2;
9StatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_9/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_9/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������2,
*StatefulPartitionedCall/mnist/fc_9/BiasAdd�
'StatefulPartitionedCall/mnist/fc_9/ReluRelu3StatefulPartitionedCall/mnist/fc_9/BiasAdd:output:0*
T0*,
_output_shapes
:����������2)
'StatefulPartitionedCall/mnist/fc_9/Relu�
9StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2;
9StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims/dim�
5StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims
ExpandDims5StatefulPartitionedCall/mnist/fc_9/Relu:activations:0BStatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������27
5StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims�
PStatefulPartitionedCall/mnist/fc_10/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2R
PStatefulPartitionedCall/mnist/fc_10/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer�
PStatefulPartitionedCall/mnist/fc_10/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose>StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims:output:0YStatefulPartitionedCall/mnist/fc_10/conv1d-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������2R
PStatefulPartitionedCall/mnist/fc_10/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�
7StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1Const*&
_output_shapes
:*
dtype0*�
value�B�*���L�r������IW��%��SL�x��=��_=��ս��^�1�ǽ��v�+�ڽ_>�sѽ���V����#���=�����*��g/�wֻ�k;�|��t$���:b����`��n���<p#h=��V���!={: =��T��O��/�>q�<�=�=�柽���;��W��}�==�� ��p����a�;�I�<��N�����X=����h�����b���U���8���|=�{8<>��<6�_�oP<��%�ϊ�<s��k��R���̾?=��2����V׼RY�=�Y?�A�=��7=�.]�N�༻�O=�k�=� )�qT��g��r�î�=1 <(K����x=��]����:OU>��>��!=�C�=3����=&f2=T�<��<�!8�?G=�i���6H=1$�������#��B=�O�O��97��nx>�{F��A��<-t߽��>ɇ��$�IgU<����gn<e�=���QQ�;G�5���<G�<�z�@K���V�%��;qX���袽{7������i4���u��s���d�彐�@�!��;P)���`�N�6����m���$n�ߎ>>���:��=g���ʝ��V������ �ɽ���<��"=�0����E<s����>U�Z3�	i>T����J(�������=N��<|Ц;²���2�F>���79�M�����JEC�?���ʷ[�#C<
+��������!�0c�;01�=�}r=�턽C{���ѐ�M�ҽ��*俽�H@=�����E�qS�y(����=�"�<�*="� ������ͼ��ʼ?���_�Ot���Y'�\'���ƻj���� n|<�RA��y�=JU��p��LԽ�����#��z2<�P����]�l5��T������<���=����e=Hq�;H��h6�<�ޥ��6������鮀�Zm��z.�<3f��↾-�L���8=im>�?��i�=3�j=����׫ԻB��=��^��[�=�*�=Pw��F�={Sź�o����;V�>t�R����=D.��X�=�E�<w��=U�7=ӭ�=��=�Ž��|=��߻�y�<\���7Y(=Е!=Na�="փ=�߽�v=^1��š�W�����<��O=��A��sj����1��� UB=���<a{���J�=�S=M%�=�/�=2>"��ꭽlHý7�G����<�Ȗ1�����j�<�I�=GH�*�S���S�z�H=O+�<�f=�Q#:�)��iﺬSE=��K<$Ka�n����v���eQ�F�=�������Q	�є5<2�;�bz����*s�=[^=�K���v<°�<$��<��.=c���B�x=�_����=[���o��^з=<���u�]���=��e�	Hн��-=�pg���ݽ�`k�@{����e�<�Fս��_�͘'��;��}=�c�=�����z=0�\=���=�>�;3�\'�6_��2+��/�5rK��c��b�=g�-=c������a��c�8�jH�����<Є��2D^��A�����r�ѻ��E�A�>�.�oH�����=��<d��=��S�豸=:�=\O�=�b��i�;
;}�W�w뭽�s=��>������9���PU>4m뽲�{=�q���r=�sB=*?��-F�!�[<�\�<��޽�N��.�<���=0p>��O<�eq�L�<��=Nn�=��=�g�<%`U<d��9���(�<Z��e�ҽDK�=���\8�=z%������Y��<��4��n����F=^m<�o;��3.���s���
<���4��q�a��(��2ĻBM<�(�=�^�J��;��,=ro����=��T=�ͽ�=���;�M=�ѕ<�@Q=C�Ὃ}��c>�F��=�_���2>]	1>n�����p��<����W�����;&��'���7 �Z�Y�!��s���y=� �<yL=Wt7>ɳ	=�+ؼ�,��G�!�-=6���3=�x�a�(�T�������C�E=���$�L�0��=���=Q�=���&-��i��=dj�=g#��)��=;8�=�	�� h<�ӽ�Y�=�j< k<��=>��; <��=̘�xn�=��;.M4��5J�;?5��Kt�>@�������p�K$=t�~=H�=>m�<�ʈ=��=�>̟�<"G�<�]��Q����j�= �<(�P<�IR==���x���u��s���3ߚ�Y?�<��>��E:pz�=�I#�&��>x�	����=�� R�=�6��H�E��q�>"�nܽ��n=�aP�=�o��Ձ�<�E�\,L<v�H<&��<biy�H�,<5w=��=\Y<F�����<���>׼���،>�=�e��<���`=p�<�a'=ݘ�<������/=��۽�j���T	�G��=�4h�
�>)��=a�<3]��H=�vd=�eD<�s�<�=B�;�6�w����/j=cݼ��>�Kp��*E�=�xս�=��߽����/�=<�^���s=(?�_�G��l��mm�9�D=�8�;�����Z��󮔽���=�=�ޓ�_�'=�J
���<��.<�C<�i�= 5�<h؉��q<����&>�J��/��=�R,=&��ԏٽ�K�Y��R�Ѽ����l�^Un=0ټ�3��;�98ca��a:�Xv�9�>흄=�߽&I���=�<G�]E�=q(���&�:�/=��o�8�ؼ�YG�����*tڼ:M���V<�y=;D�=$�#����<�ۃ�{�J<y���=1|=��H�I2�=�c��鬍=h6�=x)�|��1S"�a���%�ɼ*�}=.���==�K<��<m��R}�ސ�<dQ�/2`=��f<���<1z�-c�=�S� &=����=��V��:룽�~��~�Z=�X�vr��B��=���Q���h<ߟ/>��(>��S�S�����=������%9̼�`�<��x�y��j1 �w�|12�Yv�ʯ��2g8=2�1=Z/<,5��j���=��'��㼼\x	=z�Z�%�u�ø��'�D�vp���G�;�>29
7StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1�
*StatefulPartitionedCall/mnist/fc_10/conv1dConv2DTStatefulPartitionedCall/mnist/fc_10/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0@StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
data_formatNCHW*
paddingVALID*
strides
2,
*StatefulPartitionedCall/mnist/fc_10/conv1d�
RStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2T
RStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
RStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose3StatefulPartitionedCall/mnist/fc_10/conv1d:output:0[StatefulPartitionedCall/mnist/fc_10/conv1d-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������2T
RStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
2StatefulPartitionedCall/mnist/fc_10/conv1d/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������24
2StatefulPartitionedCall/mnist/fc_10/conv1d/Squeeze�
:StatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOpConst*
_output_shapes
:*
dtype0*U
valueLBJ*@6�s=��!���=	*=�=�ý�����<��7-R� (�������,a=L��ț��O0�=2<
:StatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOp�
+StatefulPartitionedCall/mnist/fc_10/BiasAddBiasAdd;StatefulPartitionedCall/mnist/fc_10/conv1d/Squeeze:output:0CStatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������2-
+StatefulPartitionedCall/mnist/fc_10/BiasAdd�
(StatefulPartitionedCall/mnist/fc_10/ReluRelu4StatefulPartitionedCall/mnist/fc_10/BiasAdd:output:0*
T0*,
_output_shapes
:����������2*
(StatefulPartitionedCall/mnist/fc_10/Relu�
,StatefulPartitionedCall/mnist/fc_11/IdentityIdentity6StatefulPartitionedCall/mnist/fc_10/Relu:activations:0*
T0*,
_output_shapes
:����������2.
,StatefulPartitionedCall/mnist/fc_11/Identity�
2StatefulPartitionedCall/mnist/fc_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2StatefulPartitionedCall/mnist/fc_12/ExpandDims/dim�
.StatefulPartitionedCall/mnist/fc_12/ExpandDims
ExpandDims5StatefulPartitionedCall/mnist/fc_11/Identity:output:0;StatefulPartitionedCall/mnist/fc_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������20
.StatefulPartitionedCall/mnist/fc_12/ExpandDims�
QStatefulPartitionedCall/mnist/fc_12/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2S
QStatefulPartitionedCall/mnist/fc_12/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizer�
QStatefulPartitionedCall/mnist/fc_12/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose7StatefulPartitionedCall/mnist/fc_12/ExpandDims:output:0ZStatefulPartitionedCall/mnist/fc_12/MaxPool-0-PermConstNHWCToNCHW-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*0
_output_shapes
:����������2S
QStatefulPartitionedCall/mnist/fc_12/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
+StatefulPartitionedCall/mnist/fc_12/MaxPoolMaxPoolUStatefulPartitionedCall/mnist/fc_12/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*/
_output_shapes
:���������^*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2-
+StatefulPartitionedCall/mnist/fc_12/MaxPool�
SStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizerConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:*
dtype0*%
valueB"             2U
SStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizer�
SStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose4StatefulPartitionedCall/mnist/fc_12/MaxPool:output:0\StatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-PermConstNCHWToNHWC-LayoutOptimizer:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*/
_output_shapes
:���������^2U
SStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
+StatefulPartitionedCall/mnist/fc_12/SqueezeSqueezeWStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*+
_output_shapes
:���������^*
squeeze_dims
2-
+StatefulPartitionedCall/mnist/fc_12/Squeeze�
(StatefulPartitionedCall/mnist/fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2*
(StatefulPartitionedCall/mnist/fc13/Const�
*StatefulPartitionedCall/mnist/fc13/ReshapeReshape4StatefulPartitionedCall/mnist/fc_12/Squeeze:output:01StatefulPartitionedCall/mnist/fc13/Const:output:0*
T0*(
_output_shapes
:����������2,
*StatefulPartitionedCall/mnist/fc13/Reshape��
:StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOpConst*
_output_shapes
:	�
*
dtype0*��
value��B��	�
*��Af=�ች�L�<3�����M<z���@>�g�v��L�=M<1ؼXl��.=��\���=aQ�aS�18�eA`�t�#:���=�n��_3��Poӹ�Ԗ���=&!s<ֻû�T�=��<�L����8^=��<�)h�`�B;��<�=i�0=kf^<�F���7�'�h=5%=�W=��K<�B:$�:�����󖻨�M=*嚼"�)=D�v��Q<�h/=Y�.����<�uP���Լ� ���3��q�A=A�<F+k=�'=R�Lb=V9��zH�.�<�(����o�<��t=-�G�|��<�o��#�ؼ�y�8�5;&2<s=�p��1�V�>h<U6�c����V��x�<��&=2�m=�x]<p��;�f7��ZN<DU�H�����=��c;��j�Ks�y��:B�n��:��;��Q���o�|�C���<b1�=9^-=V���J�=��:@�:]'��%��L=�{=�ֈ�L��<�D��'=��ƌ�b$�<t6{���/� ��=(��<tI� Ň=>43������껑�i=���<		=8FF=�+�O��<E�X=0.=k�<�� =�k;��O�C�K=X3�J�<�0=D���_o:=�G*�e�W��<��G��1��i=�R'<4�������b�:(��x3-�����=�-��6t=��e��hۼ�9�=�Zt���)��������<�	��P�����Q�<��ټƼ����<��z�vu=X\c�V��HHt��<�<�`j=K!D=�B��]�Z�����6�R�L<S1˻�����(���_<���t��L���u�-��<�Ҳ���p=��<�r�<���f�Q=�헼�M��ɛ1��E=��0�|3>��{ټo��[�˼�N��d��<�ł=*a�I(W�s=��|��'��a��	��Ny��+�6�T="�f�"Y�G��<�e�<���B�G=�<Q�aD6='uͽq&=7x)=}?�<nI/�,��=�[I�om�<���<ڙ�L��2U���`�Ε/�чH��M=d��<&�<�]D��2�����;:m���I�:��R�8�`=b��=�P�;��|�:4)�<y����;�� =��ؽ'��� �t���Y��>�FT�<�;2=O��6U%�k睼��=�Rr<S:���4�Vt=�.<XJ�<)���O�<搨����;��¼�̼�l=q(��;�X���'�������.�_��<@{e�b��'�>�8=����q�<<�O3�p�:�B
=ְ%<賽��~a�d�2����<�� =L�<r�<��-�����D��L���<�ރ���(������n��<n���j=��g�7�0<�G�<�'=_Q1=qʸ:�@���J�<)���:�<!���5���v-=��U����&�;(�=��$=�[G�\٠���_��Yռ��5=�Uͼ����d|<Ξ+=!O<�}`��Hb�� ��<8�ۼ8�Y=��= Ĳ;<E� �����1=Qtm=&�;��>��WB=�z
������&=g}������m�<;x=��<0���
7�׿;�OX=X�E�Œ���	=�)=�q;<(�<V�Լ
�=l��<�W+<hL;��t-���P=���=J<�P���<�_���H=���������h-=l�x=u�;�c$�����#�<��r=l6=l��=2Uȼ�"�<I����	��D�<�� �t(`�1^�[��]p�;3-�<#߹��4c�+�=4��=�켵����8c/�<�;<�+{��=���=O�7=� W=�v4�^�%<M\ <B��=�ʽ͹c��烽��ż��;.�=��<,���A>̽�� ���'��yg<Mt�G6��ƃ<��;LcS���q=�=9`s��\�;�ϼZL�;lH+����=�7�gѯ<L�b���t_��]�H<�,�=ڄ ��q�=���#��&�л�쩼�`�<���=��m��O�=�����b�5��=�U<�m8=��l�=  �;�%0�=. ��腽'�=V6����Z=緡;"ˡ����;�DD:xE἖�o����t��<�Q��������	��sֽ�[ =;+>�0�.J�I�;��e=�m�;��������֝���2<����+�V=���[Nị��<0=(����;�%<�����<�튼��6��%���;�)l�=�9�<�(=U����gӻrl�<�n�;#7��0t��=��0=%Sv�[�Y�>�׽a2�N���H0���c�����<����� �n���ͽ�S�w0 >�!w��a�<���pi�;�x�<�/A=/����;�tt;��M�<�`f��\0=��Q=QH=�J�=KME�i�l�.�����ͽ�~�<��[6��-<½y{�=�=��<���u�R�%���?=-���˶���?�S%'=NTT�ӈv�:eܼ����S�=l0�=蓼�D=��B��
�;&�i;�����\��+�Y�:���sd�=:+�<�\�����}�Լ�/�<��D��r�<��@=�O��;���i�{G�馎;��=��5_=)���MG�c	=n��<8����Q6<�_�=���<r�a���=^����@�Y�Ƚ�|�� &z�o�m����<�*\�o�6����P���^]���=/��;�)�;�K�<��^=��<J�<=5����4��魼T[P��"�<1C�<f��<���=0N��hP�=��<a?��`F$<�r;�(�<&g��o0�;���C��;��ང`���FB�-F�=��6>n=3=�wM=�P����U���<
�ݼ��;�/���눽���<��P<Nlv<5��<,�J���e=���Np������*�D�>����!��𭅻/b�;�3�ڐH���K��A=�#�<��_=Z"�8�<�>�<�!u=�J�6��<ĺ��'j���=l�
�/���W�<��C��:#=�ɼiػ׵<)h��2��_t���A��1�:��<<m=�#��$S���`�5�����<+�B����ɺɽ&�I��-�=�b�70���J)�7���ڍ�ʅļg�T��]$<a��<m8����?��<\�-���?�w'���]�s��<��(=�����==I �;��>���<�ۆ��B<���LyϼEN{�C�<}�s�j��<�/�f+c�2�g�0�¼��,=_�G�& =<p��<�4<,���>I晽Kƽ���<%B[�Y���Wo=��'=f�<���a=m�8�2㼇�=b�|֧��䞽{f�����4!�-����S�<��ʼ�h���D����=|Z+=��^<wm�l�:=:�½D$x=�<��s+��m��<���;h����%�<,��;X'׽���=s���g��j=��߽�<<=-���s�=�3u��&�<tbc��=��J�< ��(mü�J�=g��<�.��gj�!'ּ^�6=fػՍ�<�1����^i��6X���1��X=X�=Ia�@�Y=WR�9:(=�1���e<+~<����SO�iW�-NZ=hJ=!�+��e�����<�t�=�D��
�K���#=�7��>w�H��<n=Qռ���-��"���<���;"xR:K�S<���V3�=V&=���YE���<����8����4<,�=a���8=�6�<�s��ܽX��<�#�>aL��;x-�:e#�����=�y�<��E���Z�%����8��%=�1�	%Ἡ�����=<%�I�r��	��@8�=-����=R=�����w��Fn;�.
��a����U�,J�<�qk��4Ի�g����=p�,��r=�ZE�����ב<���;��!=�J�������p䟽�����o=�s�����;��ټM'�<��ǽ\{��]dg=\_+�F�=��JL7���5��؁=6��=ș���ru�|�%���KI=��;24���ꌼu���Os=&�9��01��HMm=�
�ҷ<򆡽L�T�������:l�rN=�H��B�줎<�����BR�w=?DH=|�k��{=���'��H����<�Oϼ�q�;8b���{=�$��9�ۋ��<2�!�=E9<����l�v��9��B@=|q0�[/��MI�<֔�2%ӻ7���/�=�5 =z8���Y+���
�:���n �� 6<=@�=�W�����o=}8,�� �1��|�<�uý��;�ӽџ��[�}=�?�=]��jTn�*L��>0�Q���8����i��S���0���Ż�L3=	���<9�(�Xٲ=9���h����~=4+(����<�Nͽ����ʼbŞ��@�=�=.��<<A�Lv����g<���a]꽸�<0��e
=���쫷��=�>�=�}=�u������b��H㮽Z�=,Yj�g9���_��N<ja;�~��< ��Px6:����a�<���:hu/�jS/=y�4=�L��x=���L�n��:��\<=�� 㽼%g����˽�=:��=��W��=�<*�<���8B=��t=l��	Y<b�ܽ�ɽ�<�x��+<ĻF�:T�ؼ[�f�<㍽�>=D�漖a<��<$n���|=�@����<�7N<��4=������;�ǽt8˼iK���=H<^����=��-=�w��a<���g󓽝�A=O��= n
�6�(=[��7�W�Ƚ3 �=Z+e=�>�<m|��޽	��f.�%�O�['��0�<@A=�Z�;=�����㨼��߽/�;w��䰀��[ �����I��<ʴo<�ĽV嶽�x�a�I<��4=o���
z��sf=���<n=I�/=�;�_�sK=�9;�e0�Ƽ�eF�R�?�8�]�����	=K��<��P��_{���gk�b�J�a�&���K=�i<V=�<J�
�5F���Ƽ�4��P�=��r�������v=�W':�}e�ڐμ�c���<����u�����-={�E=b {�Iw��G�<m�Ž��i=�o��W=����t�ͼe��;�ѼgQ,��Ȣ����<�K =������:��i(=�����Y,��}�;���<!@}=�
7=�]z��qV=�IJ9%(߼�?B=���*=_L�=���>�=^$�<��};��X=9~0��"� �=�ͫ<R�x�?3���T��f��=<����%�e$*<�νK(7�Њ4��ך��	Q���m=�z��w�r��a{���<הU�{=�����<΢/��)=<�
_��J���<<�Gɻ;���
<G��� ;3j=���<9�W�	I'=iA�=�����I����&	���ӕ=�N��m��o�<�=��\\<9ź0T�< ���׭���D�}3�O�9�M=�.�Vm���:�8<�X�=n�n<�z��h�7Uؼ�F�<'�A<�K׼C����:���=�1}�9�1<��c�%�u�����z�W=Ct9<�=�����<\=��';�R�>��QW�<��.��=��5=�~��9f�=�`�<!/�/	 �W�=쏙<]Ps=h�o	��(��Us�="]<�k�<��=v�<K԰�{G�=Bĸ�/�(���F�mf`=N����*{�Xy��=+�=^�U�2�F�3m<=��½�d=�s�=�Lr�܂����ļ��.<�.=���P=X���k�<�=�,��Y�=�	#;�h=Q���x~�����=e�m���Dʼ���<?h�<��ֽP�-=�[&��t�=�س�3�-�>�;�<��.�!���u�W<U��u����c=[��=�4 �o�q�ݐ<=ܜ��_�=���<�yH=�Dh�b�D�X��;�4=~�]<%.y�+�����c<�޽v�<U-w����<�]��>����
��̐p�i�������ש��ѽ/>;��j ��}��`$�FC�<�;��=�)}=F̾<�߲��
8=u�D��d=Mrz=Ʉ'�^�<�����*���b��-u=��=��;��.�<�b�<��<�!���(����< =��^u=yT�<�tg<D(��f��=ջ=��<*:���\{W<� v=��ɻ��(�<A��<�hͽ��X<�<nj�=x�z�H���`�=Ǆ������=�\����}<g���g�|���h=Tj��46z�P�<yȧ<,i��JP�����DIf=e���t=A]T��∽P�=l�h������o�b��Ǫ��#��ъ�Z�:�c�X��W���f=>��h�o<�ƽ5��P[<������劽Ѽ����O=�}��-�<&w��*�9�����kL���L��ER�t?�=�ؼՠ��U�r�<�<�F��� =�nԼ�=�)"��r��4�����t@�<�Þ��e�<�@�K7�/�='����������p��� b;T��<rn�G��]�ٻ�TB=����y�d��<=��=�r�<1ꅼ�j������ �=�F�</'~=��R�T8�:�m�ѷ����9�ܘ.������&�w�C=\�H=��9��.c=����^j<�阽��$=2�=��N��,X�<.���U�=�$�vɽ���R8��C:�a��=�5q=�ŷ=Ԥ޻q�@����=��|L;-��P0<�f
=����!�^��Α(���	<EQ�<+����=��n=>���"����k����E�u���U����ȼ�����<΄|:횼��D< &=/ʉ�������L"������F��e�=k�Z��i9��P��ٸ;Wk�=b�ʽ��Y�	Ү<�(��n�<�pZ=LFq=���=�r����Z<<[�:*��=���&��d����)�D�<��м�S����2z���;����=H����F�<<%���L����qo�A�|�N�ļ�#p�]	��H=.��<��e=��L���$�=3�j?�eCA<�2���:�<j�k�=�z3=�X<���<�I�=�aǺ5�A=� W;��ּW�<���=�;�ٺ)
���h���+=d9=MmG���C=dkI�/���Jo��%<�+���ӻZ;��ܽ�v�7!� ��+�< ������<_2F<��~=�B~=�1�=�x�<ڝx=^2C���/���
>�(S��R�=k`^�"�<���wE����w=[ū=>8�=̤���{ =,���$�=����w:<�+�A ���=�;���<bʰ�`�W;�y'�Bw�q�� �$=�Q[=��<�V#�FI�<%˼���<l�$�;:��P���Ť��ֽK�*=���=�V�=F��v��ZԽ��>����4�=gH�<	<M��|p���Խ��g=�W�o�D��)t���ӽm����>��к�=��#�GX��=��<��(=�*�={X��թ=W�0���=�J=7뻼�<�=�΋�tb=
��=���.�<��8<B�����<Y4=F�=��;e�_���Խˡ�������<������� ��<����u�a��:[��:Wـ<���
R�<]�Q�J'�3����y�="D��z=���;b��=?+]���=�����*��O�����=n�����������1
<)b=�e=�v�;#��<���l{��i�ݼ���<�k�=�ٻ|?]��;'�軸/�=k�ͽ���=`5�D�"���H=}�<qܩ=4���<V[¼+x>��e;½��乔O<t(����ѽ:�L=M�'��6�=S����-l<�ջV���󰖼��
=s'�;�a=�w����=�X=��F=}���P��<ēo��(r;�����xͽ���=�d��lb��7T�F�9=kP=-$w��7I<�^M<�������<�=�=�Ғ;$|=��=�8=�=v���_4�o`�<�J3��>>=a<X=`�����</��="y=�L���41<�m��o��=冽��h;U�i��Zҽv2�<� =����!�/=.D=
�=_II=ޒ?��-Y�M�o��4s=�>=��!�ڇ9=]tE���C��b̼�= ��<i�h��o=��:��՗�8�<�c�=3K�7��pC<H�r=!o�=X�n=!4���c<��<Rۜ�`������� ����#=�j�ԃ�f7=�\=�O�{Ƃ�Kr��Lz���p˽J;=#��=�邽��<�O�<��2��l=DuH��婽�rp=�8�������s<-�j��K%>5����I<��8�s�,;��O�7�%�6Z=֍����<Q*�=;�W<������̝�,{�<�������;􍶽f�+��?/��fy�+$}<����Jżl,<��;��k��}�=;7=Z��<�@�:�Q��������<=IϘ�B����e����<��r����I�Ľ��K=��Ż+0"=�ޞ<�����I����=k� <3X�=�⺽��<urN��4=wk=�_�=��=�����<u��<�'��=�<&'�x�-;��"��Q�=�]���ڼ�R���Ͻ79 =�H=<�0<j5���A���[�s�:��n=��-��=�H<�ک��-��r=�S=�	<�������]<0�=(m���!���a��Zr;<��3����<��=�K���!�h?��&-[<�]�=5�;<���<NL�<^�L=��_=��P�<���� ���=#O�=pm��+ =�ȼ<R�&=A߽}�S�9�F�3;5�9u�=�Cd��;�=3�r�1
¼G���=��=!�ǽ���<0�"�K�ҽ�Ę�uI =e�����Խ)��:�<�=�hp��^��E��%T<���<�$ȼ%^�<%�h� ����<�!(<)��x�<�g�<%�=� н��J�L�=@�x<$YI<M�:�g������dD�+�<^�p;�y-=#3R;*k=0�$<]ü\2̻�����LL��3�Wm<�i�Qb���W��4��=�߮��x���>-�e�h�4<�x�<~����:�ϼ�t�<��<H�Jan�6G�=\�u��Y$�l::�:؊~=��>dh_�]�=k��1y+��w�=0>���L�=0BB�@����2M=��<��b=�ȁ��0��U�v+�<��!��71=�0�<���=�����<����~���\��<!�.=�.;�P=�@<�V��=��G���C�����'.�=����g
��rݽ��;Ƃ�=D%�%T1<�R�<��=�����v�<3��	$���$<A��=�t=�ˍ�(��ެ���U�<�7�:'��3��<I���|�<�|��o���:1Ȝ=a'�=HD�=/��g�,����,��<x��{]�<�{=�ִ=�cL�O���6� _ �4��7|�=�<}��:�$����<H6V=�����<���y
�
��=��:��Y<�	�v$�� ������\�ϼcA����̦�� [�<�`;��8=�g��j�y���=М�Z�W��>x����=�Gx�'���r;'�7u;������̛Q�va>S��=]�>=S����V��K�(�;�X=��H8�IAP=�3&���=K����4)��4T�Ԥ =��e=�g��������=�rc�W��=D�<i|W��C�X^=�R-��/8�{{��[p��WZ<��"�e����)���~r�S��̰����[V)=Ž	S��A�g���=��=��K=���;�*Q��λ��=@�7=�~����;r�s=��C=�=7���$Ε�����<�q�=��<TG<�W�=s�=5਽24<d�+��ߘ<q��ý�Ω�Ɵ�=
�ּߖ4�J]V=݇�=[ ���lh<1�H=C��=�-�;�ч�F<�4��x�
�5�׊�=���g�<��=~>?����4�qK½ �񽥊-=B�B�@sJ=�60</��=bX�=}�=}̳�@��=^b����7٨��7�ʳ"�4ȇ=�囼m�=7�s<�Y����
�%��;:�;����c=�u�= �*=%<�<b�b�@�J<��<�{�0"Ž7?�<Zt<�����a��t�=M5��]_�����;�=�¤<�T� �A;Z:<>e��U]r=?��M]�<�*���RU=!d3=�N��2Z�<��&=�f����=�|�О@���
�x'=���=?�:���=~���!��=������������7�>�===���<'̶�����Ո�=U#ý8^=�x���$���o=8��h=1��<Su[�"�=�xO=�Z¼�צ<���26�g��`��O�=']+=�	�<e�9=8I��Iq�@�J=�����==M/��Q=+�Ҽ>�\=��<��=���<O�<)Gv=�]�m�<���>����c:=�aN�0������#G=���9	��Y�Ƽ��]�-���OU$<�f��缧Ɛ=/=���0x��������;z�T�OK�;��ͻ����,=� =���=Tj0�D�üy7�;��$���L=O4w��G�<�dR=d�_�����h�񘔽�ۓ=�5�=�o��亂�����V�u=a����z�<��i<:�c=|Ʌ<j��ƫ;)��=!�<n7����V=�X���#�<� �=�q����������<�{�W�P���/=��<��<�Up<g�ҽ�4���O���;�]=lm<U��<��=�`�<d�<ZW�"�=�@�=��V�:\`j��_��㙽�^a�Z��I�\�v5;j�6=�^<�2=X�ŽT�̼ʣ�=7���F�L�;�A=���H/����<��*��OY=o��<�[x�o����1>2_���X=`���r�U<�aQ��O����}��;,"�}i<��=i��!_���;��e���q��eSy�F�1�kĭ<��ӽ��#�ǒ	�J�x��&u=y�=̇���bM=p���UN�<�6=���;��;�#�<��;<�����=�L�=|��=�0}k=�
G��,V=�0���:0�������1��vw�j�&��4�)?�=�{�=��=����}<������<mV��]w}= �]<'��˩�����)_���/�޼D��9��=�����+ս�=��=�Q>��=��	���\�?<z=S�D�s���%~����;z?w�g�=�KF��g�;nE~=��Ѽ�c�v�a;����<u�l��X;=�}��}ud=�@ܼD(�/ڜ�]F���x�=���z��*`�<�� =�ʼ�G[==�=c�K�&�_���&������O㼜�=mDX����<2�r�Tjj<���;	y=�@q=����,ߔ<��<��2:��=Zp׼���<A^$=$���`�$;�d�<�';���=�;)�UP�<��@��`�<J[�;*��׋��J��5�6=<L�<�v����<�$��f'=�t�=��q;�t˼ig=�N�=���=Ζ����<m��h=	��<�i_��̽�ox=H�N=��<�
l�g'<���i=��TǓ��!��{���G>�����R����;PS<i����<��U��f��6<��E��%�_,�<����+��f�<L��;�]K��m=Vc=|%�;�� <����=�<��GLk�,w�췵�ʘi=v6'=��=��.�=82����<�3�_��=�Y���`=ѽ�C�=��<���<��̽h�2���=��<?o�ro=���;��9=0&��5P �1�����;bȖ<��>xĎ��_ <�����%E=Sk)=�3��BD��w=��ν\�n=G�<��<����>���=�����Ƚ�eY��8�=KټVݔ�Q>��K��l|=�н����x���"� �s��8=`�=�����=@h���=�k=�Ð<6X�<�ax=(��;��{����;G��<���=��뽳-
=�-���/���+���9ż�:f��Bu�<za=ۡV��Zk=YK��Ӊ�?�J:<\��=o*{=j�=�
�<��M=������,X��wX�7��<2ǽ�}�t��:F�=�� ���=࡞�� v=S���l<Pc$�\O�=�g�;�V=[!��*=�U�<��٩���K�Y�8���	>�%=��<Ԇɽ���=�Ľd1�<������=-)!��S����q�ƒ�;+Kϼ�y< �������
=�W=�	�<pȜ=.�a>�=�fW��ռ�zj=B�u���q���	=PH+=J[���%�t�f=�,�;��9���ڻ�炽&��瞌=^�;4W�=zw< "�=����˕��	��qB׼�Ѽ*u����»�=�i�d�=��~��y5=���;�wX�V����=�ft=�B�S�?�Wũ<�6=$���9�
6K��B�<�_Y<�'��JZd��@�=N�@�b�3�7üJ�==?C<[�n<��μ�zi�p��<�Ԩ=I�ݽ�݄�p�X<��;��R<L�"=q�|=�Ε=��7�<@�=�=G���;�ݛ���[��J�T�==��,�~^%>'4-�N����f��ԭ=��(�%�;��<鋹<6XW��� =�p�=�i�<|�����Q����¼�X<d6a�GAɼ:=��4<>�=X���oz=iŭ�m:���P�<U��<e�9�ݓ6=~���ʂ=Ǵ�=J�E�� �<��l=W�>�=0�:%4J���<�����P�����d��`�<��<�I=i�����[=��Z9�0Z=c�ɽ
����z=0Յ<�6�)�=~�=�q2=�3�$��]����m=�X�=JaM�x�������=Q�{=�1��!9<��H<%����7H�x! ���=�NY��pK<�3�w�<�Ӽ�2��@8U�Y:ü&|=��-�	=l;=�����N=��̼��8=�Os=� �;�Ui=`��<JNZ=0��01�=�/��(�=�	=��<n>��l2�7��=ɳ��K�Ƽ����o��$'�=9�=�u=��;���;�/"�h�ʼa�u���<�I�;hl�<�ԯ=[��=-&z�n�����<�U��*1>��!=iM>�)ӽ���=%j���=�5��s�����<9�!�܅�=����vM�.}�s��^'(���;�1������a鯼��=[�߼~1I=�1�٭�<Fh��>��d}��J��W��<e��6��=]�~=异=#э<�U��ob<�<}
��Q���0s�� =�l+<�=u���ڤ���f==��=�?�6�S���8<)&�<����� ����漉_�<�G<�=�7�iY<|J�<��=7Ƚ���<D= ����ݸ�3�U=+�l�C�y�bc��3;��=9{�=���������7=S�;!��9ַ;�z
=�=~�μ�=�߼To��9���<�C��L�>N=~�A��%Y=��]�Y� ��T<eI�=ݴ�<��8Pd�=��;=v(=��<d��<.ٗ�OX����҇<��Y��b�|�ϻ������<I���BL@=���wAr����<߯m=��Ȼ�܈��]�Fy7<�H�܈�<�v]=xn��ZƼ���<\ٽ�
[�r$��Y=u��i�ｎ=A�0�L=�%�=���^a�(��=��`=�)#�o^s���_=�p�<���<��=�W�:��;T��[k��-἗%�����q9=V�<=��g;��P=��ܽ�=n(�D��qօ��=A0��>��Σ��,��w��=�F>�ټ��{�:)��Ht=Ւc��.=�s�<�]�<��2���T<x��<$穼��'���==pr�y?�<w��;A����Q��ƶ����/=���<�q=�^ۼ�k����^�y=�>Ľ�ԛ=#��j��;�h���=���_��<�y�<����P��U1�=����;��4=�*�<``�<Ґ|<�U<���=�=�R=���=����Լc��;#�l=؄5=�e
�9?�=���=F�x+�=���G�=�.T��/G��=Z����꒽Z�N>��,<l�=�R����P<=ɽ,7�����=`��l ��}dH>!/,�!�����Q=Tw�<p�:�uյ;��<����w�<�ݱ=�Lؼ���<x���=r��=n�`����=T����e�� h>_~<-7G�o����m��|M��� �H�R=��5�l��<�\Ҽ��#��4���,y=r�I�Y�uQC����.��=f��<$r,��
�<͞�=�k�=�ɛ�1�����L�,��=�\k���]B����<,�<|�<�2�;���Q��7���6	=�Ȱ<iK=1P�<�!�=�B<�(%������K�=Ǽ�;Gϼbq�<��d���@�p�7=��z�#���>G;���#.���>X=q\���z�;^ͽ��;�S5�1�3=�c��/��q��6�~�B��=x�<>�!,�����B�>�U�<!�^�����<�;<��&�ظӽ�E$=��=r41=��T<�آ<P�	�!�E�`�Ϸ\Ą���=��^���=K�̽`��<�����<,4��NI������I�;���<��=)I���Ud�X�����?<�<=��1�ct�;)5"�%��<T1<V�
& ��g���a{;W�;&t�<v�-���G�h�x<;�)<��ռ��g=0�����;鐗=&2v<�o=�{��e��e����0�?
0�طK�� 3��wZ=�AW="��<,w����߽5��<˙f:�s�;GC��s���:L��I�<����eC��F���9��n
��Ў=ki=��<��p=تN=��<�M<nF=~�%���
;�ˋ<z>Ѽ�Q=M��/����&�����Y=#��<F�=�$=d~�����:�!�=SV�;'��(4<���.��=ʚټ�3��	ҽ����k=%'���=��=�ߍ���'<����Ǥ[��t�=��=�}@�4b=R =x�P�P˔������I"�[ô<�:�{�=�S|=�@��w�s��Z��w�
>�S�����9���ۼ�QĻ徭<񺀼�2��@+���"��.ǽ��X���;ӊ=w�6=����
�K��'����`�<��`<M�<��;�D�r<�۷�R��<'��<��S�̼��D=�����ը��`�Z-���C#=c{�<Ìͽh|=�۞��,�=�����]=O�A�n杽���<�%齵3���� �>=$�<��,=>�m=�o��)
=�Þ�Ó�<nM=|�<+�'��W��|R�<V�B�G�;I�3;9��<֐�<	�s���?��We�"~=ǩ�p�����
=^U!���^��[R==]l=?��=e�;û<��=Y���z�=�2��㣽I|�=���=��=�Ũ;��s=����Z�<��<2�"<=��<�=���=�; ӽ�b=Ӧa=��D�Ͳ<4O��(�-<��g�-�=X�=�����~��;G�:�n�;]T =�;'h�����<�9�	�;�O��s�<I�<�;=�И�\�����K=u�
���n<򷽽���l!=ѳb<�%o�1�޽"���E��M=�L�<�f���m��K[<5`#�y��:��D=T�� �ݽ�^�;l�=�ȳ�Q�<`��<�I����=7�C�َ��5;�w9>L9�=Z<= >��A�^=���=��=���j=�h=���<1}=�O=a��=	���;�=�;,�2���'=X���0@@����P�g�4@��y#�R���%y=��=-���V����1꺼ؼ�P�<˄<���g��$�=܉"���J=:�<sc��Z�V=[�'=�{ <�V�<���轻���	���I���< �� 5q:���<�v�=r�.�
�'�z��=M��<�����p�;��<Xٽ&S��X�_��A�<f��#4x��{�{��=h�ϼ*�o���u��2<���ϯ����L>�ϕ�	�μ�[6�P�{<[�<�A =��
�/��<x��=t�>�X7�X=mG�I��<�ۥ����V"k=�0�<*<,�~>a���NH=ٌ=�+=�-ͽ��~=qJ�<t�"�M����2ɻk�c=�=����ρڼ[�h<D1�<������{�iF<=��==�R?���[���>=���-eS=���=�B��e���G��Y����->Px�<���z=z���^|��ý/=K\�8@���vh=k�N>C��=x6=����In=�b�ڼ#�&�s@�<_�<)�B=�*<�sS<�<�-�i�=��~<V%�<R���¬��鶽��=cF���v<�b-=��H=00 �fq]=��(=�H �2I���Q=i�<A:�=�ƫ�Rr4�DFk;�t/<H�d<��=_FK�� O�3RV�*;=�p��t��p�+=Ϸ �M8U�J��<��=IUZ<M�@=���˓]<Z��<fd}=�k�㢽<s�+��81<OY�}u�}/�<�<j�
}��X�<��;��j��������<�7+< ��<7�����<�{ټ\�1�.t�=z�s=�l=�I��0��o��=�瑼�u���4=�g��pY=I6��O˼��޽{�Q=J������IQ=�����B�d��\=[&>��8<��ռ����C�:?s.�7{�E����2��E =��	>le��Oc�=0�s;���U��7��;���<���)=~ג�ˇ����ٺ�f>��H�<v�v=c�=�`C=Ѳ0��𦼭�)�~�=C޽3=��=�G:�b���>=�N2=�4�<Q�`�'Q=�ؾ��=��<tTI���ٽ�.�;��wP�4K�{�>��=���<�w���\=���=kMG=��#�=�V=&M�;8��=9Ș�.N��cF={�=i}U�iو=ÜS=�$��H�Fj��L=_V �`#>=�_?=/{0� ZG�x��<mٮ��R(����=�1e<՜o<̰^=$%+�U�����E����μ8=�㥼_nŽ����)�m� wݽ80�<O��`�=��;.�C=�Q��bs�=/n=f��%sy=$�r��B� ̾�Y�X�RB��X�%>c�<�`m=k��<�j�{;k-�<�:��\�����z��~�=��f p����<o�(=s'<����>�^�����{���h�<�@���:=.*\;��2���̼߈[��2\c;�7=|��)0;�u|�^ɦ<.��?ۚ<{�@�����r�L��b���l�X��=�^һ��4�ċ�=���1�����;�)��*=k���9�<��0���;��=�Ų=E�6�<��<��<�y�#ou=�)�;�:��1�='������ �	=�⑼?�><�&�<�[�:`;�<��/=�L=W��<�C������:�=�1�<��&=�\�G��Gl��2�j�$*O<���Z�l=L��<�ߘ�׹���MW=y���]W=B9��6��=t�U=�C�����==� ߽+j¼��b��)�<�H|��R�=�׼�-l<�*ӻ9�q=���k썽i�%�`�Ｔ�#��i#�y��)��=Dٻ[PG=���ŗ�<^�����%��pܫ<�v(=��,=��%�V>���<� ��*�=%D�=ϋ< �< @O���<=U���̽�i��}=��a�;Q�a��'=�<U^�����Z.�@�ȼ�g;=ZQ_�X��<�O���;k�=��ڽ$��`�J�N�n=�ü۸ڽ2#�<��r=�~=�k�=�6���e=n�8��w�<�w�k�����)v;�{=�:мYa��?�=9y^�������H=GP=�l�=䉨�(ٌ��)���2�;2f�<ݓ�<˱R������F8=<-�<�짽;�=��@:*<v��C[���|��{�<�>O:	�>��Y-��f��ߙ%=#�S�
�;;��˽T#����$=c���Y��;�d=.�ļ�=�h��$���i��l*=�3�'.F��>��$�O=A�<*C
>N��YV�=t��I3�<:�F���<%�,=j��<���<3��P���5�����q�;��B=�3��V�Y=Մ��=]8�Ga<���<X��|}�pP�+�����=��-=r�=2��� �M� �3���1=��<SR߽ zB�{�X��Ѽ�S�*ha��}���o"��#.>��]�=�C���=S�j��*�<���ic�<:�z=K���Df�;�(���� =���4�=v� ;�+���(=>=�<m�=�*=�S��kĥ�� ���V�l8��k=�`>������6���e��u',<���R?a=�E�k��=�*���μ�#�:�L��
&�����=І�=�6)=aT&=eC�>s�<L;��脃=v����U:�;�<�5�=��Ǽ#t�����>Cݺݟ��Q�a�=��|G=OF'�{|K��`�</ݼ�==�� ����<�F��+=����Дm=2�l�T/˼���=��_�P��=;�1���b�1>޽�����0=���A��~�=�_ݼ��C=ɝ<�t=���<����,��Kb��إ��,�=�'��
=��N���W=�9���Ƕ<;�ս�<��;�;��>�A������ܴ��@�<�;����;*�=qMz��M��oj��'�$<�X#�S3�<�v�:���ez�;��B��CX���;nX<=~Y=�I=쀽g�����H������ֻ<�y+=��f= ʏ�U:=@���8�F�=�p>_�����=c����K;�3'���2=+>��������򧽝�>R��=()���#��
�=C�u<�a:���<�D�<��<7{*���S�1�@=̉����<�ʼf*���ס=��޼ߦ$��2�=Ml|=:܀=;3�<�B����<�^�<Fe�]?��#���C8>�=�z���:���
�+�=���R4j=���;��ƽ��ӽÛZ��p�<�Iw=�J�= "�;��=�!�<�;�Z0��I�������E�Q�S�� k=���̼��B;��Ƽ��3������*�/}�����m	>�5�;ϕǽ�ę=�Bֽ(�x�e��hh���R���ʽ��=�v�{뀼S�5=;f<^�<P:g< �Y<��%<� �A����C=�4=>��E*=l���8��U���������?1�=��(=��<:��=�!���&�]9=���T�X=�� �!1�κ:q�p=�������7��$g=�!�=������=�Z��>Y����y�=j���6<� :=n��;V�=J�`=�*���<'k<�:�mt�;\�=�_\<�6=�],=��ѻ[]��]������oY<���=��<�˜<�����u=|ջ�=Ds&=j1U��g���m��f�ntG>�(%=�)d����8/Y=v���͜=���{ʪ��);=_>=}�o<v��=ޙ'�~�e����?�'���нb�58��N�0���P=	$2<(����N��};Җ�=r�7� -C�X潮j�=�g�͗>��������|�~�=F���Hk���{�<��~=3�<��T�/��g��=xC=��T��H�����;���&&��^�ɼ�����Uk=��Y=E��<�{˼|d=p)i<Yf˼(��<P��<S��=��f�?½�,�:B|=�>&�`�;���Ǽ�=�>���P2��	~��8�<v:d����<�k���Ї���=.%z<�~��5��Y�T�wO��\����=PԤ;�=`O� :�=���<_/��D�<&�N:7.��1p��c�<�>�=�z�ei���$��o�X��My�w��=�,�<G��� <�k�=���< ���r�<�v�*�����=�j�=�a�����=NIU�vYǻ쪀��8R�⮽	���9��=K�q<���w=�t=t�3�p�ɼK�=_�&���<�ѕ=��1�m��<��/=M�<�:$�׸����|=n�2��m[<��f�N�V=B��<\�������<T0��H��褍��5|��Y=V�l=���������:Լ툡���k=���=��=�ᇽ!%���_��Ԡ=�o��r���%B�uc1=�M�_�=�����=u����\��{�����K��R�=�YW:4��<Z����&���<Z�8=za"<s��ʟ�9a�C�f�|�[��H��<��O>�@�<�!�;��>�ż�^q��u�����=΅'�)N���x����L��=�=""Ӽr�����<�2z<�B=1��t����L���Mc�pS<a�"=&ߟ<�U�<�
=�u��c>���v<~�ٽh�=;!�<��f�;��<�s�=_��<H�>='E>��~�=:�b�D��<�ꣽ�>=��;=ץ`��T��ǁ=ֽ��+<���ew�=n��< :�i�v������=��4=�Ȉ=��@<�q:���;ƞ(�R����ٽ�Xp��S��/:���G<?����=�m=��^=.��܂�o����(9=N��<�蹸�⼵�$<������9��W����4=�ZQ�L�W=�o=dɪ�P�*�C�K���5�'<��$#f�S�3�t����=���=M��=!��Q���X̼���<���ȣ#�5D�=X���0��<���;����^ɻ<�}��ʲ�C`��Y�����<"&��0=�<}{�<J��"�={1��yy<q�y=:�սb�,=�4S=2� ����~��=pP�[�<�r������ǽ���=��ּ����yT���=�ږ=��=0��<۰F��Ѐ=��켁+H���== ��;��ػ�CŻ=��r=[������n=��i���.=�ͷ;���<�Ӆ=(�.�̯��ɤ;����氜���<��4�y��=����� 8>Z0����ޫ�<�������eߟ<�R%=��<��>;�� =�˔=[:��VY�=;( =7.�f��<�� =V4�`���t�=;{�<Ų�<₅�c|=�l�=�`�`%�<V,D;9+ ����_4{���!=�O=;J���ༀ#���_��<�=����'й=���<·A=,F��Gkɼ37=�ƅ=��c��p��|����l�<4� �4�.�T= ��S��F0�"�<���ԧ<w�e=/�=��<�ps=ϡ�=�1�=(]�<-��Ұ���2��S�[�q=��½�g=��~���9=HR�=��`����#��Q'��T�=SB�|)A<:�=�w<Ո�<�=J=ɽ�`+<�f����:p��������=������<-�Y�_0)��e=K�;�'�<ڒ�<��+<+�<�D����=����h`�;�Rm���I��6�I�p<n�.�>�@��<�״��֩��-�=��=��>����O�ս��üV��;��O��:A<�b�:����G�=6|���$��9|�I<4���B<U���ҽ1�\����=�^�;C*=u��=��\=�f=��<|�==
>-�� L`�,��;V�=���:;�h�WT`��-<��V�V#��*X=��V=�Σ<��ƾ�=��-<31p;]MS�w�=x"�ަ����Y�<I��2y��2=���;�߽�㿽����NF��\�(�k��>d����0=�A'>��ὒ,.=�A�<N�����<�׀����;�<��=�3���q=��;=��=V�S<�r=U��;�����h�=��D</+���k�=_U,�ȴ��mU���*��Iг=f
;���������=����Ħ�1�����Y��<�;�=<i*<�K!�J�(=·�x�����̍:n��=�ݼ�|T=��ѼeJ]=N��<HHd<α=��>=tHo=�bɽ$�L�+��=5d=��c�6�k=��8�a��<�=N�J	¼����� �<4K�;��=�M���`�<Ŭm�g�s��)=��������;t��� ��i=0.v=�s;�1<:BN<��� C����{xm=FNҼꅽ�l=T=��^L�<ծF:�<3������=��N=
�{9�����̽X/����1���>`��=����!d$��'2�q�>���d,&�ۣ=鱶=��U�Ѹ;�0��qXo=m����=����u����ǻ�9Ƚ�y9�|�	=����Q�<�Z�=�9�=Pj=��E���a�=���=�T�<1U��ҁ=� �(��=���vaz���ɼع#�L=��q�9��; `�=Xl,=6|<d��jJ���@μ�Uw=���=>%<���'Q6�@���X�<'�2������c=0�=;7�;8<�<��]�7�<!Rǻ��ɼ�fƼa���
��=�)=O��t��7w�;C�o�4%ٽ��=�~���%L<	p�=,��=����׫�;������<ug<"�����;�V��ag�;�1l� ��=��Y����0�Ӽp#=��v��BH��	�����;6�4=�ˣ<$��<�<��/;>S�t���N���X���:���c���t=ʢ��ڪּ�on��l0=���6[�<W$9=�K*��@�<Ԍ4��(q��<�ݝ���z=��a���#�BҶ;�xͻ�x�=:����y�J�=#�Ӽ�=?���#�gB==A�3=:����9���Ӽԡ�<�9=H4#=�t�<>�)=!i!=`A�Ø�d8K��`%>P)�=D�V�ڦS������E=����A������PS1��l��C��"/�r:�<����>�����ʽ��A��H=t�I���k]/;�{=��n޼��;+wڼI�"�`\���t�=��=њ&�yS���=z�=���0��l� ��=3�������w<W"�"8��V��=DQ_������>K�������CӼf�|=�+�=G\�=�J���1��|���=��=�wF=b��=����6�=��н��=9�!;�b�<F�\<��O�<�@<wb�=�n<�=x罽�ގ<�J��K�){��������;�J�<\��7��=M�~��=8<�ռ.�����μ���;J|=�m�=k��=>��=V(=���Y���ɽ4]==Jӽ����v�<�����yǽ�e+;�m��m˂�ȡ+=�� R<��Tv�����IlX<�7�=H��]���<v�O�Ϯ�= ��:�kJ=jW��D>Z'=T���Yn�#���ҼS�=7�,�v<,�;�� ��c)<�t ��/C���8=ħt��5:��-�<��B�`s��׹��7P@=���/K����';R��<U=��[=0��������s=�z=��μ��<�H<��j��4��u+=��;4���v� Ng��]�<pP�<[�d�}�> �ѽUI�<����I~���V���ZG�.�ԼL�=�콆Ϋ��E�< ����YG�-֭�,����M����=�ր���1=�=x���֠���a�={i�=��k=ֺ�=�J��"�̽�����u�=�GC��ظ)����(��q��;� z<�K4=a	=�9=�V�E �<�7��˺�����%TQ����=T \���k��3�="|����=A	p�s۽�3:=�}׽�I�"3���m=�#��H/w���i�^�|����tRI�oѼ;�@>�_�=����=���:�z�(�</v{�ԛ�R������=炏�_s�=a,�X	��W�ἉA��T=�.�� �!8H=[b�={*�<���\��=t=���O����#�[������ϯ=#����1l=���:�[�<�Lf<�l���C=L ?=I��<@��V_%;ŏ�<�&���G=F`��w��<5)���7;e�����`��r=�W@���=�)<�;%�~�b�[�<�w�<�z�<�l�ptQ=Oe;s��<\|=�׻�0:�6�_<̱t=�1=����o�6�~=�-�����=��J��X�%�L=d�V�Q�ۼ�n�Ǫk�
}���E[��)�<jhP�oC�=0y��j=ą潡��=`�>���e��������ѽd�=��἖N���"���b�������=0(=�t\��J��� �UW~����<���<�,�;�D�T�Q9Tm=�9<�sY��M_<��=�)>;�{�a�!��!��,����f��𣹾!�����=*7�;���;��)=��0���ӽ8���F>��B�<iv�;�Ľ�	k��R���JE=�R�=�"��?+�#�=�.�9������$�<����Hk����,=�����>m�b=�=�l3Z�����d���N�f�˼�W�=X�=c �=�ϼ#ni�w�=�ѩ<����b��v�����<j���Z��<_��<Qa������''=��ﻛZ�;&D�;�&�j��ot�=2��׼� ��E��O�<_����9?=(f�=bM��[�M=�Rh<��{I=Ʀ���v�ƀ<�=d�g��jm=�gH�Al�=lx��r��=���<�4���������Y��=��<E�=��<��V=��R�&���	��6~C<�߁=�<��@�='w_=L���`��<����n���,�<2�<��P��T<�n<r4<�䶁�#]�<y.���D�Y�_�>�#=�5�<�C��<3��<�h�ܠ�<Թ=�y=�	��=�����]�=�9��AXU�0������>��=�6B��<�L�4��ώ/=&���ay =9�=|φ9���<��;�S.�o �t�O=�w���=%a=�)����m�⃾����>`A<CW��feN���m=x��=ep����=)^=�佸�꽼���oz��qi�n'O����<��=V�%=�m�=*6�j梼G�<�N<b����t.=U%�<ɗ�<�~��@Vr<�J�(�,;�'�<os�<��<�eb�N+A=�9�<C�=y�a�� �<J`���$=p����Y=�s�ۓ=�ܼm,�<�ҽ,�F��ū<^�=v=v#h=o�<�s=՞=,��=q�������g���.ͻ.~����\=�Ѽ	���?�<�hͽ����P=3����<�wn<�]"�i�3������W�=z�L;��{<t�ԽJ Q���=��$�/z�<B��:���<��=��!=����$<[Z��,9�=.6Y�����4�=�
˽8��=��(=Y�;��ӽZ�A�sxd=ꇖ���㼞{,�p�Һ�ݺ�`��=�e���'��4H=zgi���<�h�=~���ѕ�<z��<
�v=5��D��; �=�ױ<�� �������j�Y�Xy�=��d=��������~�;K����g�v���,�)/��+�"�I0��>��=?����+E�<�G�<��mK@�q݁�Kf� c<K�=���<>�"=�ڐ�Vȩ����Zm�<���=++.=Y�z=Ǳ�=RO������P��=�hȽ�+:<�&���,�#�6=�
'>���<�Tֽ�<)����I= V��Ý=�AP���M�0���<�G����=%e=H��d�$<o��L;X���Mx����2h�;'�;f҉=P��<
ux:�)ۼZ�;����w�l��;��=��f�HB�=�{�=Y�;�6���n�<������=<�.b=��=rA+=f+0��v���2��:��(��?9�=fǖ�m�����>D =S���7<�'��M�~�v*���<	.r=��<=��Ɓ�<���gC�<o2=O�6=���:/��h�M=��e������<^�;�X>�Y);Я=N�o��S@�H�	������	���>����]�`�ZĈ=G�<H9�=�A�<�5H��W���Jʻ�"8�#�J;"�{��������	=4kq���O��<ܹ�=�nR>�8���4=T2����<��O����#�=a8�<����c��?�O;=�"��������Ҭ���*�\�
>1r�=�舾�0C�M�{���t�<=@_b����A��Q�;૶�W�=�U���m���̼��⽌�r�t�B��9�='��=�To�F��ߒ�;�=H�-�k��"'<��=��4=�D�����<�ꁾ��=w�����h�{�����L½jP<�=C��<�LS=�˽���<�I���b=�N����I���%߼�!$<vZ�=@�=�D��L�<�����ڽ�����L�#�\g
>C��=��!>%��=����5��<�L<7�ƽtV��!^	<j�=%c�<��v=F��<mG�<�S���:�����rTY�q�3=���<��Ӽ��=*0=VG%�
�=��<e|���<֡<=���:�.�q�=A2E���I]��� 
�#}�<
�ۼ�{9=l�(��~A<t�=�h��gd=���wn���Q=�y��:B=��:��<�$�=�ޘ<A��� ���.׻6Hj=��=��==� >�p=�X����������'���0�4=���=����Uq�<�	�=���=��I�=H�<K����
X=�(S<�ɻ��;�-b�B�^=�a�<�o=.��<��<��M� �'�nNM���)=L�*��v	=�]�t=��=��z���;�ύM=$t�;rz��KCS���=����H$%=~ �=)��<k|罹j�=�Gi�p�><�Ot��)��� <A�	=�;����6<YQ3;M&<=�̋����<��.� X��̔Q�h�=�V�=+��}5=�f1A=1��=��佶֚=�n=�Q��r˼���=�N=�>���^�=��n��Ƿ=�S��R�}ha�����$�s=�=JF<���z<�:��<>+{<k ��@;Y�nh�i�н�4���I4�@�<��=_���b�ȼ��=�}H����<Oq��}�=��}=��[=��8=��̼�ґ=��f��?ͽ?�<L�������<���p �<!��<rmE=5	�~w��1�����ԦD=�� ��������GO�Ɏ�=?���sA��=��u=}����=pmۼ��ݼ-3��խ_=�f	=��v�_.=���<�\=ߐ۽��C:}���<w��^�;/���FD9=���,):�#��f�v�i��=zO��R�_������׽[���C�y=k}=�w�=����0�4�i�A=	=(�;=������JW<da���=����,�+���/;}@*�*��9��.=#�k�U��$�:6!�;�O<�{\����<q��<,�T=6=�S߽ؕp��ɗ=�N�<��8�%
V>= w��ǯ�S�����`�������;��뼛�I�������=,D̽���]h�����=������=u�6=�+=�����E��%�Dz&���=���=ɿ�<�T���=K�=�r��	0=5��=�dP�V��XWܼ���='����<]�����:y=�"�<��X<��+:>�y���CoF����{&��~&Z=� �yw�z��n/=��=���,�__�<+<8�Ҽ�g��o��r��x�{;��/��zټ���Q`���3=GY=��=��0<}���\�O~�=����}�;���tE=���=��w���O7�溱�W=��=)E���1=�Tz��<���~�眜�cl�=.�'>Ӽ��";㺼�qż~�	=n�;{�-=�zk=o}	�۳D����=�{ ���>>�6�}�*������
��T]�<�PμLSi��>n�M��C>�F���3��޽�B�<t^f=�Cu�(�I��ߋ��<I������!�<_s�=�p�:OȎ=��"���:A �̡��7m��y�v������<Nս�e=��>1!(=���<G@�;�3=r�<��%=�b
0<�gJ�k�R�E˖<�H[���k��-P�H�R��t�<Ӱ�����~�&��==4*ݽC��<D��B.�=��=���=�'�O�Q�yw����q�'3���=��.-����<�G�<���<�=c�>�i��?�
e=��=���<���< nѼVP�˕�<C<�< �>i-ɽ��(�f"�ܥ%<r==9�=p2~<�=v���?]ý�n��v�D=� =&*�SZ��<���?ܙ=~U=sqŽf|�=v����<��A1�^p��dM�<�x=B�,=��$=��<�6��ف=׹���1��.�P�.$L<�%�ӎ>�,�����=fX�ƒ���=ݽ��/=bt�=Q-�=�˃�=�=�ȼ�bp<lࢽ4��ഽiP�=}=�9)<P͝��q9<�g�=&*�����;A��DTg����=e*=Ƴ<�gʼ��=e������<!=��-��J�=O�F������-[='�H=�t8����=�u���X�����;��|��/9���5U=��=���=ހ�<s.����<~�?��5j��=���<��H:�ƨ=�CH�(�|��֤���O����<�?��E���������<Y�@0=��<,�[� =�{�;��4k]=K%�<9�%���=r[ݽ����q�=�$=͔���=b�>&�f�j��*Ӝ�!��U��<LP<�;[�����xE��M��1�{=}�=E���F�:�:�"�������[D�<\<WT��K@ü"�G=;��]s8����`�>�U�<��Y����=:e<ɻ�^~���f�tW,=r��y�=9��=F}'=���<!���I�=�L=W3��G�=qk佤$r���׽���=b?=(�Z����S�}�:7��=�T6�fFJ��I���ͼd�<<�H�Wg@>����<��T�;��q�;�'@���=�V5�oM�= +	��#�<J�h�{B��9�=��<<�����OI=��w�K��8���~s=��=�bC���
�x�h.ڼC�3��}�^�5��ڻŦ�ޙ�;�`k��m;�ס<�D�1M;�� <�{N�A��<P��=�f��<�=�t@�Zb{�?ڻ�ŏ=�����;�ݥ�Ul�=G����=��?=܃X<ޞ��j����=�Q�B�0��h�<0�+�4y��ŕr�L7Y������w����.=롚��:^�Ƙ�=��	�O�Y���G<�mo=O�u�5�9=-�<�2��VBռDѼQ��j�!��g��=�+��b
=��e���;��r=
:��H=/Nݽ��@<L�=H*˼G����="�2�����(b=ņi���r��O�Kt�=x3���8��ys=c��٥�礰����<�_�g(S�~����8=y������=F�
��[=��=հ�=����{&�w�A�̛��<t�-=�>-$U�V��=e�<YE��|��=2_=޸�< ��=�y�������c�S��r@���l�m:�;��j���9�|��;n�\<G���=�+P�n�<�d%<���;<o�='�g=[�@���нÐ	=� ;=^<���L�[��=M��=�/t<d	��)�������p�=d~=��~���<^ik=���=,��=2L\=�/f<��G��}<ɏ���=c�L=�U�<��=��漄�=�3�������� ��	c��7��[{�<R3��{�9���D�1p<��%=q��gX=��M�
={X�!.��L�
�� ���μ�����=m�9<����<h�bi	���
> ��<Yo4�+�����S?<Q=����O?�=������=iH����};�?b�n]x�Y_�<o����W,�KnF����= ֽ��g���,�U%�<�P�_IV=O���G3�=�-�<�h=��w��;�<�
p�:HI���+演�ݻ�R#=k͆��۽\F>ー����>��R�˥��+���=����׽ j�����=��:��>�>BbU�`iͽ.Le<񀆼G�
��^-��~K��9*=��$�p�t��)�<"=��=����y<;�y�<�NP����rp�;\�z�b�߽�Y�<���=�*˽��ǼS䗽f̜�usO����==5>2򱼳��=�T��W=d����i=O��_�*���@�	��;_)�;LP4=�ta��fz�{��t ׼|e==�@=Ƹ���=F]W��Տ=N��=A?k�s<�d�<*��<'�E��
^�&<輬��=�?�<��y=����h��𲼪,�<�6=c��<"���'��<d��F�MŽ��B=���=�$���R
>:�;;�z��W>91;���)�S:I�R���;�"�V/n�$�=�i��@��Hͮ<�?��w�=�!�t�=br�<�Խk�)�>��B�����th=]�=�l��V�Z>K�O���Z��4���=t�<�h�M�-�������gb�=k|{=4=�"c����=Ѐ�<��<T�h=�����=�a����*�!D`=@�=OA���Aw<��N=�޽�id=���<3��<m#��Sx��t=gjE�T:O�c�<R���J.�j�<�z"��8�弨��v��=�,j>���=y��O_T=1�]=�Ar�c2ȼ}b=W=f=�V�;#��<�L=��U=�ٳ<S%�<w"Ƚ�q�<߀��̆2=7����I=����ךp=;��;�T���a���>6�f<�0� 8t;IR���Mn=z{Խ3ň��8�=T���ݛ�e�=���)1K=�Ɣ�PG���I�<���x�=sO����b3��=�!=�᪼��V=�A<NYz���`=7`�=�e=��J�=J�½_����]�W��<��=�ƫ=~i����C	:�!�Lѽ���=���P���b=�Du=�v���,�;��<��M=�Y����,<d�
=`C�;Є�;w�	����8�(�ʽ�]=�O���j=�T<��_=�~�� �}=%^�<�������;�ƽk�|��f���e�<�h���+k��[='�U=�s<2���
�����N�=��=�%�=��;
�ƽ4�=�Qn=+u#�w�����>�?ծ��CE=�F�<Fo6=^Yb<�C�<��=�����<��l+�=���9j�w=�s���|�L/�=�ݒ=�`ݽa�y=�<0��=iaI=�ȥ��Y����v=�x�=�6(=��B<��=�zc=�p�<b�E��!%<}��#�:���-�h�?�άy[=�ϒ= 䩽c�=1�~=��S=��<g�)�Ř ������a�=��<Lv���
>5�)���ѼOvP��عH�}��ӑ���O<S3�<�w�=��:��z��x����=��G���i�J+e�0�;��;��p]ؽoG<=�=�7��<�9$���><��>�6m�M2��{�b��">!���gf�<������<r�ۼ��"��5���-<JO]�D���;껒ϽEd0�_
ĽIA3���~��p<tp;���Y�����h<>�=�d��w�L<��-=�@l=�C��oE=B��t{v��'=�KӼ��%��"����<�E=ߠ=�������;_V_�a�s���la�=(1Y=���<�
���Լ@�/B�<ן<(x���抽�)Z��/�=.��<�{��u�;Z�2��e�v��q�<��佦�N���;="X�<���oN&=!���e������ú�������=i�.=���=I�:�y=-\���e<P��;!��������=;��� �4=a""��肽}�ｴh�=~}O=�F<�`o��>�<�z�=��;���=n�3�Z%,�W\���P�����J�;x��<>|=�|=���<!���G�<C=	Ƽ%#X=TXQ��E��[0>�4=a���8>��a��WB�����z�޼��A�Y�<"�<9t�uK����=lvg��[�����Qd��=���[����<]�4���5=
�?=�Rȼ[$��j��@$=���;&���Z/=Cܿ<[�A�^>�VL=_/ҽ �>�4�\�=8 >~*o��� �(�ｂl��ȋ=z�뽗�'���*�X���0נ<8�Z���[��s1<=�<��a�m�ܽe1�;�l�<)�><ƿ�W��-�<)&=kD�=<�#=���i�ʽ���=���s����<\?Ӽ��#=i��A]=e#m�廈<z،�EC�<����Jg���z�'��	�;�����)=��K��_Y�Liq=4A>t꼅�<�5�������B6=�Zg���=X�w��#�Nq5;��=^%��!�P}w<WI�=���<�ޯ<?׻��ܼ1�=�������'5�<u��v��<���	թ=�����~��= V8���3�������D=U�<#oa�3{�=�$I���b��cj=�|�;��F�t��Ѫ=�c=�мs=/���Wy�=]����5�����:"=��\=�����ݷ<�n�����<����;y�ⶼ��l��#="�q�	�����<����@w=f!�<���<O>��+�=ǽ	��z��2u�CF��a��jV�<��>!������=׎���C�=���=.8��#A<�����#=�_];9�<q���~�=���NV
=15ͼ�a����򼒻 =�t�<,74�Þ:�.%�/VI=�S��h� ����=�y�=J��Qp����=A_}���<��=e*9=AI�Qa�=.Z��׋n��	�M��<� =g�=��`=�w���r��d������ʮ�<�lW��$��S)4����=zˈ��?�=��#>Yh��HQ���<����h=>�'��.��6�Q�5疽�u-�q
�;�̽�ݳ=�༮��=���2���ї�n�p���=t�ؼ�~�,��������2=u�ɽ�.�=���<�e�=.7:yW�;'�V=���>��<ҽeڼ`ˁ��t��<�y���pq=�t�<Bϻ��Q=&,r<�K��F��Y��<�6m�,�E=A�����=��^<����X��
=�%��'Y�<�]=Ly=��!��3"=���="=�����N�;�<��e���m5<�F�;�}d���J��G<�I���%<�.=@�4;|��<z+�I����<C��=H�:��R=l6�.���xco=?B��ߙh=�Q�<�~=��-:�����6�P�=M�=8ゾ)�=a�>*�ƽ�9�<��ǽ
X�d`�Se]=1�!<@̾��f⼦�]��͜<��=Z����)';�lݼ���<oC�<?�9�Y<�_;��½�'�=�S�;U�C������ߘ���v=�:'�M��=�*=��\���H��ێ=�{=�O�=)r��7��:��0�Z�z<�յ=�z�<�����=�ὀ���¼��a=�0<hC�=���ͫ}=u@=�i�B,�J�=nu���.�=B���5��$4�M�=ޅƽI��=S�����չ�lb��T�=#%漋�V=
.����)<� f���?��1
=}Zм'�j�M�2=�6Z�H\�<�����<���<�+o�jk=$�>���=W��<��c��aK�G��;�iw=�A =� �M�Q<[�ﻛ�&�� g<��h<�8�������=`�2��˼X9U=R�=�d�=��=Sv	���=�Z�<�?Y�N��k|n=d��=�m=�(m���N;���}d��Л�Q���S+�W��<���r==�d��]�<�#%��N���ҽǷ=P1�<�~�<��=��<︄�j}y;â���]Z=K/�%���a<cn���;��@=�쪼|*����<=�z#��\<����u�F=ͨC���E=��<;��<{��<������:���,�=��ż��?�"�^�
���US(>mh��^��=��ʽ�8P�����=��,�f�N=eΧ<�D=�}�={9����ؼ��=p"=�'�=,�s�	�h�4>B>�g̽θ=�lC�p#->B|����&��=���U���f� >i���G6>(��Z��1�j��,=�7���}=f��;躠;-i<ps6<;��>�<�0�<V �<4pu��<<�q�L[)=Sz=N�Y����.���ؾ;�*�;&����*��Τ0<E��:���q�1��X���4���=�;�<t$F�J5��%�,=jE�<L�=�_ؼ�5);I�Ƽ�J�<�v= ��=��*ýï�=7�Y��,�=�|��δ=1oY�LѶ�Ӷ��S�B\W=*	l�nd����
>�m4=+���O�����~=�I���R�<"�B�S�<%��<0�#��=�D��
I<cR�<�U���4���+��Wy�<�=��ͼeM�=�;֜P= =鏽�'.�0�����;CA<�l{;�y
>�9 ��$5=6�=�a��B,��x(=D���E=Ӹ����=�{.��8�
���X�Y=��x=�1b<��@����<D%=|�A=��<��V�&w5��aR�����E�=��$�~�j=Cdӽ}�����t�&�=�UżB�ҽ>�>�!��p�V�rr=�pQ=e���nC=�۽-,�=h޽�ʽj=n�m�����U�]=IR.=�G�� -5�gԊ�8���򖻇C�=���i=ś���$���g)=�f)=����1lսi��<���=�*��b;�=s�3�-ޕ<�`�L8�<e��<�Õ�m��I;^=����	�=L2k=��G<L���$5o�b�J&����5:jC�;��(=����{ 2<�p���J�77�<�ܫ=fF��g �x'���W=�e�;m�����>��O��$=f0�h=K�3I�;�w=)�G=*�W�q��<_�=]�N\]�r���%Ѐ=MŪ=I�=Η�<�e>�rQ��h_=�=0�6�闎<y7ۼ����lW� �T=')�/��=��e<Cn[�ё½�H�A�c=�=^a��9��e��(����XO<ֈ=\�[=�kY:� ���>�v<��tG����= o��#�X�\q*��q߽�¿<Ԝ�=J�<�� m=�|6��WG�ƙ��cK=2L;�����.<�S@>`�&��Y��$�<�4�6���>��ҽ̪��<»6i�_�=�=��ȻOCr<�"ȼ��+���3=S�;sRE�Ӄ,>A��P�:��=AA=�0��я*�+m��LV=��<�e=d�5=���4��<g���Z�cs=��9�û�=�ؼS�E=:���4]=L����z��v��uF=��x=�J�Ta��_�<�ɂ<��e=35�;�᧾��E=խ��˒����<���ݷ=�9�1B�<-����6��j(��N/>�����R0�2jl��g��\;�X�>a���-D�=��=o��x&=�@�=g	�����=D�7��K=�;ƽ�
��V_i=j|��r2����J��6&���<a(k����=�=���K �����!<򄎽�N=l�E��|�=�H�
>�T��Ǟ<�����O�<�n6=�v�ٍмbK|=��F���d��:�<���;�h���=
|M:�hM�h/��v%�:�bڼ��SY�<�6<��;�ס<�F��F]s�����ֈ=4,��
��n���R<}��;�
b=�h�=���=�~�kZ=
j	��v��.<�5=Ѓ�<܊�:Z�B��:�=#�����=�lM��ڼ����xt�<1��;�5��g��`��<e��4�;�ö�~%m=n2�nN9o���j���Q�<ʽ������8;<z��<];׼o�]<g�H=�0�b`#�dx�p/��E�;�_;<=
׼ ��$=PW<��N����<�5���?���>�=���<��a<���=c����;~)����=�g��e+=z��=��<��=r�<�={�q���=y��;�P�6�v� m���I��op���L>��O��_'n;)8�9k�w�X�M���0<P
.<�<&=�k���j�=�R��YOM=ݩ��ꮼ���=:=��2u�<#)&�ʇ��Qb���Jb<<�<�8&;��μ��<��ּ��F=I7����=:�$��9ؼ��=f:x�q�I;^p�;�	=r?=SC��:�<=E�<��*l=J��#Nڽ9!����=���<z=v</W=��Y�<�l�4E����h<C�J=�<7=�2=<�`�S�&=�,�=�;��U����=��Ļ8���\><��=8��<-��<��4<\u�Q��ː=PO=sj=�c�<���=Ч�4n�=���<�3��c����"m=� �;���<qn@<;�X=S$Ӽ	j�=��Žo�9�/:,��al=���]<+�Ҭ<޹ ���	���>��<Z���X�S�!�=]�n=��@=Ss�:��;D�=v��<G= QB=O.�<��=��	:V	u=�BE�:�w�:�H�԰��+�C=y��>�l��d>�X=m�}<�H�=9���1>�ʥ����=�e��d����gl�,d��$(��+�F�;ܛ���ǽ+������]������;d$����j=�޺y�:=Y=�|O�ۦX�*�ܽ9��=�>�J>N�
����=D㽢,ڽ��h<E	�̛��<f�/�{{�<,�K:�w�=|6�gG�����0�tux=;��~֔�ϏP=�ƿ�n�<��F���z=�'(���=	��xĘ����ǝ��<S�m�+�h�0��9ƣ=zL�;P |���-<��,=�'=�.1=6\=�ӽ�+=�1��Ҹ�(ؼU�<��F=�T�<D>�<zD��[�et�<�<�~����<a��=�ɋ��\�=9=�Ct�=ž��;�b�<Y����c����Z���z�=<��Q<��}=�Y��:�'����=R'u<��������=5�ͼ�{)=�V���S=Do=3�/���=x� ��,��	=�:�Y�<�V�<.u�<n䡼g����<4��=lW<=�XU�Y��Z�<�q<n�8�g�=B���O�<u�tE�蓔<�e�<��x�B�< `*>��r�R��6������W�<Ez��0?=V�<��<�!=f�&�UB�����Y�Ž�.��J�5�v�K��=���<��y�����HU�*]=��;��-�=�C�;K���C=ʗ�=3����>�=���+�������e=���
K�Jʼ=��2<q=�-����<�|j�?֎���̽��<��/<%�*<zK>��D��}w*=B�<�#�<f�6��>;���`�Q��0����������=��꼝����λ[��=(
>����6=Br����Ƚ��޽o�p�J�[��1=�lQ=�о=ݜ[�ZP���=?��E�4��:z�q��=�K=��C<5��<)���-'=��<��ƽ���='^�����=a�j��v�=R͑�N_k��x�<L���kT��-�ɼzW����<N2½0&<B#8���*=�� ����Za»n��������E��ro���C��E�:�m<Uq7��_=���F��<�c ��)����<���qy�S�X��|�bj�<7�<��սp�X��6�����=_�����=�A��v�=z����Qb�-�=��3��α���<=��=�<q���=�\�=�T�����j7�=��<�MT=�3��J�e<��=`��
�;�M�?�����S��"�;X�b=��X�3�
/���9���O=�lM=̍���熽1]�=�ʵ=D���º6=ҏ[����
<=dw���s��R�9���=��I�rT�=�l=�P���{<��,�!�W�ب�<��-� %�<W���Ɇ����u�6<@��0�=�T�<i��Ԓ �tG��^x�=���;$$�<	���R�<�s����>9F���s��d���g�=��<�8=��q�sb�;��<�l<U0�=f������<��t��i��M���<)�9����1���7=�֬��5�<RF�<�ۼ<A�`=a��K$����� �f�=��(�mk��!����;� =0�=4命�-�=|P���=�|t;�,<�/��	��=�}��=^�����<r�1;�/6=��l=j��:�ƴ���#��E+��'=��j�=@
���q>�!����=A��m�Һm���6���>���=����6��Ž�<Zm�;����wv=����=�"=)�=V^��)�s=U�0�ȏ�=�����<�1�;����ؽڷP=<��+���n�V��Pd=X�C��l�;�c;d�[=��ܼuS=n�̼����x�=1�F=���@J�=R
�5較%�������t�	</BX=��C���)���;G�A��Y�\=@��bG�m�?�~�=����M,=4Q��^�<u7���!=V=B��XzJ�?`�����Xe=g�=��A=�|���)=����M	=5Fֽ�<a��;h�<����uB���"=�������=g !>N\A�	�:Ҵ=u8����=;m�<6ԅ=����k�J�=8�\������b=�S��;��u�zp���J����=m��<#��<dh<W�ܽ1D�<��E=Ud�>h���D �]�=e��<h�<^7U�M?��g=�S�<TC�;�(<��\�=A//=�@#��Kɼ�ɇ=�
�5Fͻ�d���T=�hY=���<� �;�����mH��(f=�14�,WQ�?�
�8=�N�&I"���<t���?�������=�Q�'WF=���6>ϽJV;�淼	���F2ܽ<i�<���<?E>N�B���Q���c��4;�q=n4������ 4�;��!=�M�<u�7=�=�<�0<wʴ;ZռXj���=����c�=�A#=�MX=�P��༄?�<P���X�C��=��h�:R�=5�#�.�����g��p���"<���<�z|�.ڙ=�ݙ=Gۙ=j�
<E[V=r*o�[���a=�j�;�{�<�ć��eǼ jݺ��z��;�3��=͎k�jjf�N�h�^C�=�<�Fv�(Kh�E:w=L������w0Ͻ�����8>fp��L	���5>�-���u���4��?���Aq={2#����ʀ�9�/�=f'��:���0=�52=)�nM�]�k=��*��2�<g�<U@�<<&Z��iG�� A<58p<	_�TJ$��.=�Y=�̽?>�h��cB��#�<�A=
���R�b��6��=�×�n�G<��=��gTC;�K�<������lw����ݪ�=�X����<� ?=4g�<dY�$��<賽�◼[<�<�r�=���{����{=.�+�0�����#=���<W����8�=(z�=7�#�8m�/�`=�����3����="�Ͻ�; [�=Sb����=�4����[�Ԧ�<5
=�l4=�-��U�q8%=ЇG=���.�=��+=j�V�[�n�k����z�=�I>#
��G�;�KG�\����|*=I@\�m�Y=)�9=�~�<�A#=XK�W�<������(=�
�y���.��L$=~�S�����}���[Ѡ��2��ۼi(H=,Tϼ�n9��H >i��#���{��=I���κ��0=�s	��}L���f=��9=Z�j=$�A�>�+<�Δ�7=q�C��U��T�L<)����="� �e��܄=�ѓ�����&1�\����q���Rv<L��<����>�p��;�rV��_W=�s:ܶH;���;����R�<�4�<��g��9���=������$�<5Hս���;����ԉ��� =�*<�%Z�R�k<C^���n���=�n0�Q��=A��	�$��
�=�<tڼJ">�r�i@�������<@o��q�=��=o����a���<��;:�1;Az���(�=g��Fxg=�N<��!�T=�'�(�Ž�����O�*!�=$����=8�>:����e��A���G�K�.���M<Ct�<��i��!��u�<d P����j���`�P;�6+=Άu��Rs=�ow��X�vX�r�<'׷<�[�=��P<���<OV�:<W =����L���x��$C�<Pn_��X=� �J�l=t��<�g�=j�=#��<��<(B�gy4<D�u=��ż��=&�����!=���;hw���;-`�=8P^=Z?�=�ў�RA��w���h<�b����� qμ�La=<�=
Z8�V��!�r=�qi��h�=��@���c� @?���
=H �j�\]1��q��(�k��dp��3��yɼ�t�<��^�U��=�P�<��a����AE�=�f=|Q=,��;v�<C	 �=EB��;��;⊽t�-��/Z<��<�i�<�"*>C����*,=�������I���Y.�=X[鼪9�<eI�<Y�4���ȼ�⺼�Y�U��<W�L�g�����T�#>t��<�ؓ=y�1�<����w���W߽�&�<������y=̔u;� �=P���f���<�3�)��=�܍���˼��<�:��D�F=E��,��<�qt�	2��t�=[��w$U=�d��"�a;~ɢ=3��<=�׽���AQ>1u;����P�b�<~gk=�꒼�XA=Lֆ:fgO���μe%��0-="'=e�X;8Nh��zԼ�k=g�����[��=3�"w-���=D�6��ʛ=R�a<���my�!n�������=�t���������<����n���N1�=�����2=��;_(��[�=�=b5;=(#���<�:�<�<;����=�Z�t��ɇ<��W=uo���O;��k=~�d<�U �@�=<
2�s�=w���6��<�ݼq�[=�֤<p���E���B�=Q���k�R������"<`��<�U�;:��=���:�^��o�	>���N���~M=7�<N<�<1=��F��k�R��z=3�=����M^x�aQ�a�ʼO:�=X���X=�o���":^]��7%x=D�������q�6>�ν ���frq=�NR�0u��������_=�r�W=��t�=�wa��;y=�!�<�5�:U������=d��̕>��=j����)>�f�=���W=�콆6=[_��_ג�^{h��ɍ=��;5GG=Ti��	w���=�Z��=JQ��C�w�<u�ƽR�=-Z�<��1=���<�ƒ;1~a�6��婼�J��U:M�*�#2=�<�w}���Ǽ�#����l?�V(=τF=-S{=��
� �==�uR�a��N<5��<�[�< /<�H�=r2:����<�<2��~:0=��g��?>&Q�3'>]�=jym<��-����<#>|�(�P@M<Nt��<g�̞e=�j�x�=��=[b��m�<P����\=�;i�Y�w��<��}���=�#�<�>�9���'+�n���Z3x�d�*�Ǳ����M�!��<:R�<������A=��ļ�T$����=Oň<]f���������=~�e�hD�=ay�(���Z�����}�<�(��&�=wϼQzz=�;�=�'�Lk�<��<�V��lK���u=�YA=�o�ew�&��<�!u�O���4�>���b=��L<��~�E�I���m<�]�-�5���"=`	���8~=�8	��@�<zV�zk2>���<
��=n�_�]C�<`��LV=���=��Q�%Y��`�r���޽N�$��\*��Sм�D��R6��=�<3_= �=	�i=+�=9E=��;��H=��<ډ��((콫8X�Ć=��8�-[H�H�=��c��F<�/Q�0��=Y��=�;&=�G����c��սe�S�����z��=P�<��z=�1��i�=b9-<�d��w��煽;B<W}���/=��=�vμ+59�Ҹ�=�t��5+�����<�`ؽ�g|=�
v�%;�=�m��<��А�9�J�g��<ba�&�I�_����r��)<kT�=��u�ϓV=��C=$L�����o<�htR=��ν^�<�4ֺ�>�;�?�=��.���m�e��;���I������Ǧ滓з=(~k�I(F<e�΃?��g=���<��<�r��dY�;�s|=�}�<���<d����=���;.�'�4庹},�$+3<6���cי��=���+�=��<�Li��g=�G�n���=8ܽ�|ü�rt=c���*�`_V��k��u���nƼ�")<[2C��B�9_�w�=.�R�g�B��Ha�I���K�<$��T��=�Al�5d;MM�<,Ƚ��=�5g�~�=6P����=�jڼ~��=�)7�Z�g<��.�X�j=�׹�.��L`3�Z��(ڻ��޼J��=��R=e�=�ܓ��\ʽ�'ڽ�.U=R�=s];�v =�J��/^6=lk����;?�6=���=�ㇻrE���I=��!�i��;�||=.�A��)@=J�=O���Ϭ=%��b�1���k���=����X�7�@W�<{B��(&�<_����<�b<�k�<�����U�ǻ4�飲<��}<�%�=x�<3�=�����=��<j�ǻ	ɑ�2�<I�M=٤P�L�<��;����f�b=���=�������/Ӌ�:S=���<H��&ܳ��e�1Ȗ<���<냞�۝��䦼=UG�-�@=��c^��������g��2S=��f���м����g�=�+
��ݩ����<��_�V{�=2#=��\�愪<�*��1�=���af<<�rm<}ս}�=�BL=喻�m,=ɀI=<�+=���qp��Zv����<�P^�l�D�e`�;%3w=�Ӆ�l��=�_n=��&��a8��=��=\�<tM/���;<L=�L��ۼ��<�8O���?�<=�I�Nu�,"�8����S~=�PнPݲ;^ﹼt�6<�Q�\�B=uD%�<�d<Aܲ�p�B����.>g�=������3��j;Ϸ*��s����<�e�; ޹�. =�5y=6G;sy�<��=�����<5��=��q>%�=3H�=�uw=�s�<��9��[= ����5�	��M���st(�g�O=�n�=w��=E�I�p�=��H8=��E����<o+�I�=�VA���4=�`:;9���~�ʼؗ���=� I=WQ;���<��= M�&�<�7�=����1��m�>��=ᦻI�5��_Ƽ����\��-I�0�<�t���	=N���i���2�<8$8��#�<�I�<ʟü��+����<E�p��r=��L����=a��vQ�=4�׽���={&��ؼ����o��>��E�<=j��ܩ=��=��ȼ%<r�� <�H{�����<	�1�|�	=:%�-��=�� ��	�V8�9��<��Ϲ�߱���!�}�<�����"I=$�r�|q��Y��TV��4<�������=��0=k��\�=[BR���8��n<�J)<l��T�=����95�ϯ=�*G=o�L�ר=��;i��;�q�z�;=$�5�>={=�[=8���N=�����A=�'�+�#�8�z�冽���=�Ω=)��0�x�ov�Y�>���y�G�v[�=P!ɽ�w�����T�>�%F��>�˘=T����3=���=5�#;;
�=�fr<�x�=G;�fV@=�1<�
=�MѽWL�=<���c������=w�=���;�'&=�^��vNf=g����c�=�~��P�=����C <�-�����=�^_=����X�����C=�SV=��=L��`�O=�g
�:�B�<�!��b�l=�M ��&�<��6�Oj�=u�˻��=|t��և���7�<�i�<�>X�*=����\J��wd=�f=�l����<邱�������9=���<�[*�ԧ�<�0Y=h0��D�B�ļ�*i�+��=m�H��P۽��j=��o�GD�<����Ӹ���ߠ<�3���3=Ì�2F��5�4�$��<����=�� ���Ǽ�@���ո��8_=��0=!�Z�/���Ɣ�=�Pռ��<��qx�<2|ý4@�=!6=0S�j욽흥�J�G�!V����:��<��;n��T1Ի�Ǽ�򲼱�
<�w�<�ڡ�������5*�={*�ʱ�y":�Z#��=�V�;�Ý����=R��=�<�x=���0��d�=]�~��^���s���:
�X��^_��s6>�f漀�J�����$�=8����=��E�p�n�=��=v̐�
��f{.�� 
�l4=)c~={�z��|=r*G<r����p���D��(ҽl�=�Y7�Fʻ�C�I����(U�&Q$<��=�,���m=4r���]=���=�K=6�=��s���-������A��mǼ/2�=�W%=Ft�=�WE=�I2�� 	�3�<;Cp=eݘ�K�2�/ ����;�֎<�6=�F���"���:>a�c<[�����O��������a��<3^h���==����=1=|c�_�;���r;[u�<�5|<�����|N=�����L<g.�0�;��˼{ܐ��@2=�v��W/�����Z��%=Ŀ��l�������;=3�`�I��=y�|�m�6��Լ��0=%꽪��;�z���g=Y�<��=����B=��2=���;R�k=�B�`u�<5e�� �<�d=���xPJ<=4;����e���X�7����=T�{�*��=�);�6<��<��I=H��jے����6

=�z,�s��<��T������=��9S;����w���< ke�b��>}�Q*r�C�g=�'��j~��ӵ<cR$�R�����<	��X?�ވ�=pYļ�\<���=o��<��;w<�|Ͻ�X�;n5I����=C�"cg=�h� ��=���If2==����u<T� �!����]4�c�	=�p��;�>=�L�0A�=v 0���v�s�V=��C�.�>�)Q=�wG��l(�t�E��_3��R���aq=���H��<p�����=�m�<�G=��m�D*�;������=%3=N"�����\��e<V����D�=��<<��<���<��u=`�=0�B�)B =�-S=&��<�r�q ��8'���:�9��<�܇;�?�=���"�b��<W9�<N����=Ž���h|�;rǋ=1�\=Q=T�u����<�98=+K+�V�M��yǼ����F�S�W�A5�;x^�<����Ф���wd=V���YU=489�eX<d�=����qM��7��<��<�JE�Ylz=��𼞳ӽ@ ѼY8����ռv�=c%M���x��v�<'\��,�=o�0��6���~�n��=�n�=�o�<|��=�뽞�-=�0%���n�ആ��=(��<ܻh�B)w�=��ټu��=k�g�q����vE���G<O{�<!<��f<��3=�_ =�D=.Ǵ��B��m="S*����7-<�t�����[=�H=�	��#b5=���S�>>�Z��O=�`��r>������Da=c}���f=��I=�ۼ����ĽNu���H3��N���=aB<Hi�=��i���̻��<�m�DG&<��彆�=�U��>~���=_<ͽ�`ֽ<�\��=<�ʼ��>%6���#=0����ټO	<x�:=R�<���)2"��,��ǽ	:�=B=Uⅼ�3}�r;w�Ӽ��_= ����a|���j=R��柗=Ɩ���r<AH�;�V=���;�F��V#���\��I��.�=5�ü�O=2R@<��輘0�=h����g~<��t�$�6=���=���Z�
�i�<��<HH�=e�>�V����1�L�L��	�=��T����<B�f=5��<p{��l�G��N<+��<��ż."+����<#�o�! ���D= G�FlR;�el���˽řR�� =�ݏ�O��<��<�E�ܯ����qxܼ	<={h�<Lѽ܂���^o=G;S<C؃��EH�J�z�Jω=f��=g(��@= �=e�<Л3�̻�r=�=md����^��&L���>W������=�ȩ<"�=0�N9��v�=�<�ɂ�9�Ի %=��E�b��=�H�Bd�I��;�0k�M*>��{�=Vě;Ә�>X�5����+�Z�c��}��]�ɾ��<��u����/U�����Y���^�<��ǻ�V�=��=Q}�i�=Ho&�>��<��=L4��E�'=�Nw�L0=�	rO=H�X�ƶ�����,���<�ѯ�@t�<[G�<� +<��M��H��,V=�ŝ;n%ؼ�)���?��$���	�f<����Y��=�3뼱Ï���[����=��;1�e;��y�?��j<*K����=��=2c�=��j<<_����c��["<�=8��4�r=�H=��:=7�M��ӻQϳ����N�<��g��(���o<_�`=o3ؼ�=�R=�y�JӼ�i�=A+I��,�>���^Bo��G�<�^����<I�x�ºg��)���D��I~��bG�&VZ<��>=�⍼��Q=N�;��<��=��=?����<�`<o|�����:�x=�s��Jt��3�ֻ��.���;_2��y~$=���<(%�=��=����
 �Ʌ�;�$�*���`�R=��f=�f���ʈ<���<I�4<0ڼ��0��J*:�֍���="��
13=�k�<q�@=E��<�bJ��|)=F�>=��N���V=՞����*=��ػ�.~<��]=���;�qi=�&5=�K���5=d�s=�9��U�K=g�q��͆�=�09a:���=)�ݽm>�=\s��
O=:�9<|���G¼`>�uJ���
=�W(��(+=h�<��v=��=H�=���ڻJk�<6�V�����	=����ţA���>��$<%7<��^������F>/�o��=Ȑz��~6==!0=�K�<<}�=e�T�++=�=^���`��9�"���Ӽj�<��:�&+���м*�;�G��QX�&�������q�=BA�����ꒈ=����ꂒ;�I�=�"=֔%=K��Uء<.e��I�Һ��p����>���/�;�ソ�ݹ�F$��x�;�ʞ�4XI=� �<�}l=N�Q{b�6�7;��C��!=�=i�D=:мX�<=Ѹ����9�ʈ=i���!B���J��\(�����hS�:��<�G<b��=��8��}�=��d�죽�1q<�!�CN��-r<�^P�"y�<̴f=H<���kP��(��b=�J���� <X6	=�Gr�Z;��;�h�Oڐ=���<+6x;߰.=�U:=Q���<�pO=T��=���<�1=s�<�3�<��"�N�<���;��;b��=^ @=F)���R4��68=Yك<Ƀ�?�<-�\��s����<Rt��Z�*c.��P!��3�0��<l��<�?=�o��l4�,��<�B�9���߽/������E����<�K�=�!L�wiQ�dmS=�(N����=��S�	W#=�x�����.<�</��=�B����g�K��Y��<�A'��.1>�mm�T�N���=�$<Ëb��M">�뽉�l<R����:=;�Ȼ���<*>@�����Uм J���=�&�<)�J�Q\�9`�<��|<�C��P�=�e�X�4�=#{�<�'�=p���Tv&=dΡ��4v�3�=����N���~�<i����Q�2ކ<,�;��Tu���9<j�<�W�<���<�˦��cC=���������<��d����=>' =���A0�<4�t�8�T9��<�fr�
��<� :=2=�Yl�NE
<y`>�r��<�1r�\<#��q!=̤��^���@�Y��q�:�3�=����p�/�q�� �0�üqX=���<�y|;����E<�1O�"�=�
���㽙-�p�?�� >���O@��P�<���<Td�=t��;�M<�o+=�m缋�=t��<�����:�v���ȭ<I���z�L=����^�=8�)<ӳ��� 9�j=,�;����a=���;,�|<&5�<�q=�-3�⊠����<s�_=�A���-���2=Ծ=���)w%�'`_�E���R��=��󼈃Y=�<��i���=�%�!�ļH�=n4�= �[�2-�'=�x<�!�<p;4�K��=-l���/�;d��;�|X�>���wﲽ��������$=��<U(<[C���g=N[��|"�;asp��Ѽ<NA=j���XF3>ѩ�<��=��ǽx��=�U;f� ��&��������@:����n�D=7Ϗ�[H���C�=R�<'}9��{`�ݯ�<�Ҹ=����X^�f�������<nj�<j�s��.�<y��#Q�@�,;��=��=z�s=�f�=��=j�D��`���Ļ��<�41�(�(�Bu6�u�G�:��:�C��B=�s(��C���u�t��;��=��>H��=BL�d�B=�h3���?��&�<����mF;íѼ�q���J=�;�{q���$r<y�;V��=y+=mn�=��"�˾��M��q�=�p�;W����=�u}��o���Z�<�����=��~<�����<�y�<xռ�-�;�1w�m�`��9=oF����=����ɽF`�=�5�<w|�=;����˼d>�h��'�˼�<!��$N��mW�܊��K&<��f=ν=�q=�Rd����P�K>jO�Y؄�v��=��Խ�6�<j�'>���?�=����S�&��)��[���@Ff��@3=O%�=CG��4v~=��m=���xp�y ޻��;�U�<����ط��1���(~=���Ǘ=H�ɼ��z=#s<1����_�=j��=���<��=p�ݼ�ٽ�f�< @#=���<H!4�CG_=�Q+=�u�=FvƼ��=�E>*��;D^='��x���1��74�<�z���1<+o�<�Z@=�ۼK ��+�4�i��<J#%=H݄<���甼��=|�X=�mF�T�N�-½�F�=�=��=V=�_n<�ob�Y.=����*�.���K��ʼ����ʠ�<��;�R��(B=������X��W+��r��xl=��üC�<��<��j�*/;����e,�-��4��<��<��=Sǥ:f��l�<��$=��ӽºb=��.��>E=��F��A̽�X<m�=�xB=�	��ঽ�E�<�]��?�<_�$=RE7=��4����9�����%=�8�>Ψ�&���F�	�G<�Ǽ���:�<�<�����q�=��<�{ʽ��*;��C=�k =:!=��<��f<��=�j;����^|���%��I=�9==�V��=���K�A��W�ݽK'�="ᔼ����j=�Ѩ��d<���.��=8N˺Ӹ��b��=L�]�f�z�Y+ <�x��G�G<O�}<�E��3�<����V�o<�j˼zé<�i�=�
z�׫�����ךZ=�q�>�倾:{6����<T ��a	��l��5�=�H<�\�;���s�ͼ�-�=�@��=.���u8��H����<ӁB=�+9��B��^��ԑ���:�cѼ|K�<��.<V(�������J���m=i~F�aX�����<�0�=�<F�=ز�=���=Yӹ<�ń=�h-=�]�9l�4=S�0=F��;;Ox<-04��Ƹ=T�<��*-���:�譼+��Z���.G��'�<��n�<���d8=-�K�i��<Jm�<�Ԛ�G'��)b�<�p ;�$���!��Pr�ŏ��2e���v���;�]�=[-=z�<D6w�V�M�h擺3�;�
N�E�Q=�5�=,]��Sm)����xZg=�����߳.���\=wh�=� ��v}Q=?(f<���P�=j��{�=��4�oXd�ą��[p�E����t=Mૼ%�X��[�='�U�-ٽ����aƼ)V =���������<�&����D=l5<[с=���<��l�}�==�i���r=eYn��^�� ɾ��`��&�:U�"=��>�̈́<��8�Q�:�x%���z��"%��?���������|���}"�z>νf�=.3�<���=]W���4�f9���8��< =*�Z<�~�7�
��]��A�:�Ǆἂ5̽a�v��>�{�;=B1=�Y�����=>mZ<��=id<��+�J��yA��߽=f�=��-�^X@����6�=��ǼrC�<L#�=�-�:��R��
=u��Q#�;�-g=<�~��c=��;z+=y=v��2I#=5	B���&�ޠ�6N���;ݒ�<9�T�N��<)(��N&���M=����<6�;�/�=�E�󿷽{����5��`|=��=K(˼��}�@����]���ŋνtjD��{=T��=������<�C�����=��G=��x�������H��<�=�{�=�߬<��'����#
��� -<5��=��7<;�>=D;(>����λ�JT<������׼M:<p��4_e=���m=T�'="[���ȼ̎4=#F����ѼX�=������Hr=O8=�bX=3�ɼ�?���vY�W����i�<2B�=t��V�����P=R�<���]$��!�A��v�<��T�ap;�)���\���x��[#==M=�n5>��b<�I�=�����پ��R>'���w$z��s|��hk�גּ<����ܳ; ��Ȏ�<�aϼFRS����=�p�;-=x⏻�T����h�M!#=�t��xC=��e�m���N~9=��=gp<k�z��5�v��	��<|��=5ѹ��)�=�?=����R:<T�	�Nfd�T\��\� =�j�N1����l;S=p�m=5LX���C=E�7��E�Px�=Ȕнèz=��<u�/��w<=���=�6=��= ��'�ڽ׫=砮=:ٽ�H�=j�޻"	A� �(<�q�bQＺ.X�NW�<�d6�Y�a��qo=�	<�l��i��Q��<n�=�a�H�2�_�=����/���~:��"ʮ�3M0�P�<�Ď�0ME�P��0�	��Jd=�`�;8��=����r�-��<��<�Xr=+ �=��=��r�w��F/%�����͖9���KG=�zQ��P�W�u02=��;W9̼\_
���=�9���h=L�,=J�::�<
l =��	��..=|Y���2R<>tk=L"=��:�.=u��;h�=,y[==����<�Ѽ����]�q=��|����=���i<_�_���=�(Y��z�<���<���>=�:������<z����D=��<�q���	�"Q���K�=k�ֽ�}��-��=31Z�A$Z=-����p=6�2�4!�|כ=��½H�<G�]=Sz����6�l��<*�r�#'�=`]<VҒ���v<����r�ei�;+X<�"V�yy�=^@�=�`V� j�;I��<VJ��d��5yU=�T	=��;G�3��\.�+�����=�����\�۰�=]=�r�=�&`=���<5B=�0z=�7<�X�<�SE<�»�U"�^��<0	=؅̼�����0=����r脼�(z<x�<^.	=��=M�/��ó���D=��<��q�KԤ�[����<�hԼ�х=�L@=�/�-���s���-:<�J�\V���9=w�=;��s� =<#M��m��)��W*<�<dkW=�߅��˳<�4��zܻ�����6��P*�n�̻�H�x=�Y�u����B=��^=�)�>�@=����ƻT��1G�;D0!<7=h�:2j=�ͥ�<mE�jA ��)'�H@=�h�<��ɽ��=B�=�a*<&|/��1���;�fN<�2ռ�n��&�V�։��F=��=�=�<ë�<U�<PBs��a<#H<|B��<���	=� S=��6��Bc��i/��Z,=���<1]�<7*<��<g�ă<�BB=�i��Y1=��������=��L=9�&=�$��7������vR�uW����"�P�3;)<�S��]��<��W=�B���� �h|�����<4��<����<�=0~�v�a=��+��2b�CJ���'<pv�;ᴽ��=�����-a=xa�H�����<�\�<��N=^���~�<�� �Q�߽�e/��|=��#��(�;�$�=�e��8.h=���<?�=�p�=�� =&�A=�;	����<��<�����a=0��=`��;���^
��|�<��;��;S�D=a|�(0���@���=D�U=��L=���<1��<�釽��/=j�:g�=dt�<u��U	�=�K���#�f�<�L�=Ͻ�;P�=N���9=�Y꽋tw;f�J:5�v<q؍��<Lߞ�h�=���"2�<���X�U�Cg���<nLL=1���=���<����=�߻uh�=��z�:��<��}=�Ҭ<�U�=e��������=%�{����<����#7��=��i��=0{?<����0q���r����򼜹�<�命=A_�L�<&G<��<&F�<OF�=����Tպ��=�Ww�����Ap0=�t�<�X<6BX�2҃�*O��Iϼ��ӽ�0��Wi����<_QF<;r�<�C�;�]0��ݢ��A�=���0M�;�U�� U=�+n;�U���u��m>�#T��K =��K�{����1�=�!+�����C��?<��x�C�<=�
<�л�c<VJM���Z�M)��wu���(��x#K=��4f*<�ܡ�tdq�ѻ�����=�}ż`m���H�<v�o=�i۽��>=E��y=��h=Q����'=�[7��W=�Y�<�vm<*��<�73=�����J����Z�oN��7e=H�>�K����@�nʻ��[�6�ƽIh<A��;��ہI�>ަ=�Dͽu�=��===�Fl;_�H=1��<�)=��=ci=q���\=6�0��nq���5;;[1�C�p=�Z�YpνH\��(�<3�"�1\k=��;Ng�;6b�;z\����D���=��.���W�]�U�����at�<�'��0;J�k�0<N�=�f�<������=�Zo=wa�;��5�k��(1��%���<�^�<�]ȼ��=C�ý{Z�=��f�M�M=&��<fV<��
��C��T�<���;�<j;Q~�n���W��=<��<�<�<���E�=f.G��oX=)"��y��I�;:̅�ӫ���(<�
=f�7�cw=1����:���������`|�;�����̻�F>��h�x;>�ԑ�go��f?ռ������e������׼��<#������������D���X�=�쉽u]_=V����g)=+�����\�QD�=�������7<�Y��g/=��:����<:�=K(�=�7�VR==�s�<4�V=�c:=p�b�~m-�Kx=�Z<����U����[����=�)�7�ڝ�{���r�<D��ߕ�=?*�;�"켩�D��,l=*�9�|�X���J<<��<5� =',�:�1�<���|�@fy=ݻy�=*>R��"O=e|��<�<�i��<�e𽝱J�>À<�P�<�ǽy��<ct�<�0�̽�<A����6H��=��/=
�C�beP�����+��1:{=׉��<5)��$&<��F��h�@m�����B��Ck<	�'<��ƽNͭ�_-��
=�SL=&�.=�м�<�����a�=hP3�먽 �����l=㻠�n��=���%�{<�ڭ�G%�=?/���=ڽ9�˼`��-�<N�?�rcj=ó=c��<��X=#�r�3���W��QG<��3�=����P輮c��i�,=���<���;+<�=Z�k�Fx����9�a#=-~���o<�=�!g��=�<GD���ř���g �E6���s=	����ŻP5��;��\�i3~;�����\�텈<�"����;?�<=���ߣ]<��D=���;#��/s��P1=u	�=_���Z1>�2����9!Z�<�F��3�<�f�<򢥽��.<��������<��&�gЗ���\=cM>�Ct�=��м�f���@1=()E���=���:��<�^���jOk�����~W�<-O�)	=g��;�k�=;M=e�)<:�<̼g����̼��%<2�%��0<�?���i;�ލ�M��<ܹ~�0��<��T=�����=�b%=���7n�=7»��t<�=��;���ҫ�F��;�喽r�}��㵼(��`^'�̇�=�	�=R�ϼs{�;�< =L�*=�vƽ΄=��O���=�����ȼe��~2�;u|����<s���<�lc��9��P�G=�]�=�倽=k�=�R��$\�iQ���8!�L�=g�����<MQ�<���<$C=�'=;�>��n���l=ͩd=q/���h|=FF�=��ż�2=E��<�K����<����X�j�Q=a�=�(=�=����۠�B|�=�?�����:��-=H�H=��<Ǝ�<�L<K�����<m6��.�:��ϼ�:��t=��=�_M��B=w>ى��R�����D� ;��0�|㙻X^=IYl�ݣ=���<�H;)�;<ʈo�.����̩=�&���ɼ`�3=fA=X�9<[�y����q��<$鲼�O�:T��[y�l]v���=��=�!Խb��=�W����¼e��=��;	�"�1���m���_��oN�}i�<"�%��<g����M��ˆ���4<Yvd=���;[4B����=9�2�XVԼ��	=%��=|)ὸ	6='I����=�ॽ��;Q8<�n�:���=w9�<}�����<F/�;9=4;=��#>��J��χ�(��=>g`�LN޽Lp<�e=��ȼ������;D�;�u�=�B�� �q��0�=�`����S��<8����)����<Нȼ�1l=�}���P�<?|��U�ؽ�yM���_/���,=�k:#Η� �$�%�"=5
����=2K�W5��3�=T�Y��k;=��-���번��mm<�6"=��$��X�loB�=#T��^��_�:^��;�W�;L:A=[���5��c|9�W��=*��<]�g=~-�<l�޽�z=)�;N��=l3�vfQ��缂~*=4�W���;L��O������ve=ʊ>�/���+= �%�	؂�R�=�<��v�ʼfъ=2���Y��b(�;��^��,3=�i=uG����x;c �;��*<��~�_����`�P��f҈=la�<�!<�H˼z�ɻ������=�®=wԛ<u�⼅Î;L��8<��7=.=A�=ʿW:���?۵�'B7�P�=�����|��/)��%�;�9׹�f�;/�����qR�-����t����<@�x=|��=gzĽSW=��=i�6`>�^5�A��Ҙ�=���۽��=�F�+��\�I����=�O뼇�=8>������Ž�ü٬D=���}�ռ,=�}�<�����G�b4���z=�A:�.`��u���E��;�콚��=*TE��lT��k=Xs}���<���=����n($<eAm<�F����<��~�1=��@�E=��.<|�����ӽ�@P<����4r:�A�;�=<�b=�(=�{��p����@��J=62��������I=QT;���6�(=�7<<��ֽ�F<~Z<=:�r�w�ϻ�X=#6����`=16ǽ����~�kg4=kM���D��cq�=�F�<�h;C����<=��*=`�<_��<%/;`G=C�6�"ռ!UP=d�b� �g����x��z2����<�r�11�O��=�����i=U*E��b1��Q�<�	����d�r��]�
��MӼ�f�=�\h�	���\b�>����.=�)c��0�<e�<���]�^L1���l=>
��W�J>!��'sU<4~�V�H<�k%���6=�.v=q��ׁǽ�^Ľ*���~S�;�cw���;U�i�\Q�=8<��w��h��9Z���=Nv��l>��'=�=[�A=.'<L�G�]���&�U<�^�=*���*=bPl<�(��[%���:�xg��+�d�s��U�=qj��6�6=��;��<ϣT<��P=cP������(�<7<>�a�v�>=̲��ѫ�;\=B���)!�u�O�@�<4N���9<2,�;����[b�<b�c=�,<�*����ӽȅ[�Q�����	�D�(*�g�Y=�s�<�w ��<j^=��T���='��:t�3<��z;��=�?��S�<�cs�)�=c[���k�F�������چ���<�<�I�����);)�:�v�3���ktɽd:��V6��
=�ݽ� k�<�]D=�����=��������B;G=6~G;Ҧ�4K2��RQ���c��D��c=��I=���=0���3/��d< ��iE��.P��o˽�i0�fY���"��җ=*���@_������8�R	�;�=�~��J�=�����	�Γ{��x�@�F�1� �Ċ�I�*=~޷��g=71�=v�6<�TɽϨ�<`D�y5�����Ⱦ�>E�;Ď�=����p6=��ȼ����iL��ʕ;�ڗ�l��<zå<Qz���{�}(,=�F�<�Q%����5������_$�T�X���O=x�T�0P<�:��&�|���W=?��=p_���h�=r���AB����=�
�Q�;)O��o��=3T=B=�<��-���=�`����"�gT��?�<�fI=��{���=�`�z���bH�0+���⋽�=�ǌ������x�7�5Ut<u�����<�x���=.�꼢�A��Ԉ�I(�=�y�F�&�iM���l����\="[K��+��N����]<gQ���G������_�=����'5Q=m���ײ�=���F9������ݺ�9%����rɆ<���==%*��B���N��hA=_��6\�j����q�=ڏ�-�=��#e|<�-��O��<���<F�	;��M�1�4<d�=���y�;�hk��1=�K�F�\= y��#�<�h=yC0<���^�=d����{c�TH)�`8�/Ճ=4�~��R�=�w��哽��7����%��"���K��� ���K����<�$]=�슼|�ϼ
2i�$]'��$=�&�����=＃$v=L$X�N첼$`��M�y�&k;��<��M<�.=�L =������*<����:0��.?:�$��_��]���n��;|���-=�_"=(t���A�=Rly�!��&6��t�q= ��=a+�:o�޽�A�=��2<)<�z�=?��=�߳�G>����#ɔ��_>
�ɼxN��䛽Z���R�*��\����={;�<�;A\<~�<(�{<���*e�;"=��+�;�L����n��;:*=�������<����4�@=�iȼ�l;��"8�ƾF�0W2���"=��=�!|���輭o�<������<Q7�<��G=��!���<�����m߽# �������ż5~���(=����tB=I�O��"�;�M<bF�:삼���Z)�=]��=y=��J�8R���n=��}���5�������*�3BR�I�=�U���u�����h>=Y�м�o9<s������6���=�R=N0B<g+�<ʀ�=.���ü��<�Mv=�=�Q�<��s���!� =�L��=�G������9��|�<���"&D��F�;j�ٺI�=���2���O��i㼒1����=a9���_<�XG��|=�h��k.$�ռp��<g"|<���}˃<���7�A<�f���_�q����H=|� ���k�>c>o'����Ƚ��t=}:-�>��ㆿ����@3�aӽ���=�;�܌�2F�=R�Z<��q��Wy�-ͼ��<��=g��B���a|����R��<{</kA�g��n1<���1}�<���<.6��>T�q��<W&������G�����"<2 �<P>8=�ȼ699��T]=�?q=. ����=��1v=ju����<����_�<"�;/�=�ꅽ����]�ݽɽz<=���R��<;��;�k�ت=�!��vw<���u��<
�ؽQ�������@�<<�ӓ<�=���=��׽Һ����-��<P�g�~-���$��[�<��=˦��}2=�/�ǫ�<4���5-ʽ�)��]׺�j <�/Ѽ�O�d�Ѽ	�f���Y=�=�c�:��~V/���6��>=�������DE�{qf<��6���v<�%����<=�c=CK�_0r=�n����<贗���=���F���6\�]�O`滰3:���ؽ4��5z<2^$=�T�<�������<Dec��-=�㝽��۽�9%��f��<�%��a�l�?�<%��')=X�޽Phw�&1#�ԩ2=�����!�<�կ9����B��@?�<�ٱ=��D�ɼ��N�?
�=	u>r�R�p�ݽl<`�<.jƽ�a�<���9��,�V�p�X=	�s�T���Y~`<ę���ٽB� �gR/�r������\ʿ=�!��wQ�=�>��5=���<=�̽�w��?��=����hA8�'�<�ί��:�29=�N����C��M��=*���#gw��^ý�\<���[��~ŷ�՘��Z�\Ԋ=6]��k<���O=�FI��U�;�F��֥������@K�N2k��h�=@GI���[�IPa�K�9=��<6/p�3�-���e�!��<�N}�;zE����;��;1{o�̾\�{���M�E�κ=���<���=bvg�L2���}�f���B�� �ٽ-��;�o=�H�%=��S�����C̻���<>�����˼���`Õ�+���Й�=���y�<Ў����=VZe�Vb4�v��	�9�1߽��=5ٕ��Cɻ=�a�#�z����P�����;�X=��v��粼�i�<س><��Lf= �8�2���FG<r��VK��AԽц��S<���=Lҋ<��<�`��Ĩ<'�V<��V=�m�<l�V<,��<�T�=jg����6��F�`�齶B�[���Ek<���=lԩ:E�w=L�=ľ��i��/�=W��=Nes��A<��[��a�<��j=���B{����<Iꃽ������"���e��2=W:�=ݟ������������J�Md$��sݼ�������=���L����Mʼ�5h�葼!�=�1���$�=N)���N���r=�Ҽ8C�R�#�L��ڦ8=|?��q�l;��_�L�=ȳȽT�f;�������<�6н�ږ=<����<�δ<N�t'�SH-<���=�)y�@���ӂ����g��ż��A��Q�=�#=���<��'=���<��R��~+��ӽ���~�r=m���4o���>`rƽq."�!�e��+�����ӥ�87��V�<?	=
�e==�]�P
�lv�"���G9�hm"��(U=8�=<d`=�* =L<�D�Ľ�8���!=�Rp���¼����g�+��c��#�1�����<n�콠�=n��;ـ�<K/|<�:�<N�B=�s<��7=0��N�G�Ύ<m�զ_�1z�69���W��J{���==�)�%~=�-��j���a6 =E�i��)u�4 =�P&�_�6s�Q�=�B���ݽљ����;u����t���,�H�U���T��<Y/�<����	��D^���o<��<���n�W=�%
�32��Xʍ�d��<\0^�&��<�A���jU��G)�� c=H۩<�O�f�=d��:B��Z9�<��q<a��Ӽ]=L��=���g	=�i�!ֺ�S�=�	<=b�"=�
A�6��=
]	����O�z<��0��I�;�J��u�����~�Ӛ�=	�<U[˻�?7<��~=#�}t��*]����<!����<]/j����<���<���c�S�θ=�y�/�C��5�;D�<=��<���(��=���:`2)=��׼w�6�ӄ=�K�^Y=G<�c�=t�=���H�a� g�=��pi�,2�Д��V������ɹ��t����!�P?=�j<����;;������Q���� 
��u2��;�v*[�=$��d�=�*�<�<%F�<�.=a��`Q��;%���E�=�)��ES��fz��� ����<%�;S.<2�H���R��<�3�F=�Q��+��=޾����<f �<aM���S=���N=�<,Au; �T�(A?��}��H�;�:Z�<����֌<W��p.ؽ{5�<���əo�Ke�=x��<�"��K���"��n�5�4���\I����|>Q=��1�K=����8�5�o�<�������ݮ<�`�R0=��:�A�=T'�<v�q�_[s=	�O��䁽�'
=� �����=+��=�8<�� ���<�Ɩ=�jS�36��=&�8�aQ<��X�ɲ�=�@����A=S���ɜ:}��������L=MȼO���5f��M:m;M�6����<$M
=�	���n=�F������*���<����^�(�i�[���t�ٽ~�	��c�<84�����������K���<��<��"= ��~�9<�f��ƕּα=��=p=�}����=��\���%����⼅���N=�fߺ6�=��,���<�p�<.��	���Y��f�+����ā����a����4=%i��g�<z|@�+���\�8��=�}�����<S������d�=:��` ���d��3o�Y�=�h+����<�S��բ��؜�<Ǝy�ڹ+��
=�!<�' `���;:J�i�e_>�w��*!������������g=|�=�<�Q���=�Pa���(���D���(=�,]=r����0�s��<�����E=�ѽ輭�Y��s����Xv��v��<y�=���=2<
:StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp�
+StatefulPartitionedCall/mnist/output/MatMulMatMul3StatefulPartitionedCall/mnist/fc13/Reshape:output:0CStatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp:output:0*
T0*'
_output_shapes
:���������
2-
+StatefulPartitionedCall/mnist/output/MatMul�
;StatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOpConst*
_output_shapes
:
*
dtype0*=
value4B2
*(lN<0�b�I7=��̼��?�Q��<&����μ�Q=I<2=
;StatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOp�
,StatefulPartitionedCall/mnist/output/BiasAddBiasAdd5StatefulPartitionedCall/mnist/output/MatMul:product:0DStatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOp:output:0*
T0*'
_output_shapes
:���������
2.
,StatefulPartitionedCall/mnist/output/BiasAdd�
,StatefulPartitionedCall/mnist/output/SoftmaxSoftmax5StatefulPartitionedCall/mnist/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2.
,StatefulPartitionedCall/mnist/output/Softmax�
EStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@*
dtype0*�
value�B�@*��|>�	�<�^�<���(�����\=�j������=?.���';�ɠ=�
>I�*�)޽���=-T�=���=�OC�����nP�=�!q=�R���Ƭ=����<-ؙ=5e���Z����=i\��oW�@6v��N�=h�>��>r��8�=U��<��<
 ����>���<,)#=��=ܐ.�� �X(�������>�9�=�d�=�u�]*����A=��ټ�դ=�|�tr�=�w>�E%<��<j���f<�뼥/�=l>L��=GC���=�P�E"����=+�\=]3	>�>���=՞��4��=�@>���=.d�=YS�� ���D�xRG=������=(����6�=�
�=��>�hv�(�=`0>�/���������=���<�	�=� ���>�o�=�|��˃���=*�>�)#=~�,>�� �����+��}CN���:�]�=���=�E���n�=��=q/>q�{=��!�tP�=͎>v��=�3>�讽>�g=>t�����:b(��i�=v����͞<�s����	bZ=�q>wn$>~���LE
�m�e��>��=5�A=�o�=�9c��5>ZV>�nr�\�˽k�4<|��m�=���r
��{��n��Cy½B[>���Yƫ=!졻����>�UY=j��=���=�9H��*>76_�OL�=�h7�h�>D�F�1L3>vL����,��=�D�=�6�X=�=� =O�;�����,�=�z�c�=!=��)����=2G
EStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp�1
EStatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
: *
dtype0*�0
value�0B�0 *�0��\�|������q�֜�<z��;�a���:p'
�P�B=۰���� >O��L��L�T-<J�,�d�ؼ�Hҽ�.<��=I	��Zr	>Y7L=�{?�xH;>R����ż�D�P��- �z#�=�g�=c���)|<J�<�t�;�F��7�=S>=c߇�<&>��纺>na��ꋛ=2��=wS<p��,Z�i*�=��<�BϽ��Q<��=�|>����=��=S��Ⴝ��Z��*>���=�_=E��ڊ���㢽NK���j�Ⱪ=�ʏ=�{�<y1<>�&>��軄RX=hd ��Oս��=��Z�0��A�g�l��=T惽Q����[>��3=�G����_>������g>2<�5��t�=�y|=��=�w ��A!��н�*�C�=�{���Y<�k����<&<:������ٽ���b������	�q��{�<(��=%eF���%�����=ӫ=E��<46���/ӽ.w!<ϫ��~+��8}=ߘ�2���M>��Q�*=��=�*=�a���s��/Ҽв=I��2�Ƽ�yb=��5��U�<;.u��a4��x*�nY�=�Y�=g��<J�<�Ϻ�
	S=_T��݊#<p����cb<
�3�a�;����t�=�!T�Gj9�O�)=c<=|;�<�c���M;+_��J�<������=DƏ��f��*Z=���=��<'WT=@R廁2�<�?���UC��#�E&��X��Pl)=�==K����\��fE�oѽ��>=��o�r�佪�Q=ͭ�;wY �=��DW�<��S>�|���=�T�=Q���fk�=>��=�y�=r3�<�-��t}��_���,C�b=�԰4�>��=��=�N�5*>&6�=#�<�p�I�=t�=]/#=�I=��ǽ�޺=aŊ<��>=���<$�	�=�E6<����N�==RA=	�Լ?1��X���ܼ���<�5�<�>!�K̼Z)t;ɀ��"
<�*�QVƼ�خ�mBz���=�D�= �=BPS�Ȥ�{�	�K����<�^�=�8=c��lE��`��=�͐��;��d�n�(��-F6�K�����1��o�
�0؆�KK����:=9�a9�<��=�$�;g��=��<�x`;٪+<�\S�H�(;CG��rUe�rr�=��o��́�s]� JR�T�⽖�>��=,�Z1a>U�~��[:�rS*<.�2>��3>
��#�@<�}�=�8нT\i����]����#=��j��rn<��u��V$��1�<��=/�+���g��Fڼ[�n������=�"�=�>�<O���\�'>�����f�=f.=�+Ȼ�:�z�=���=�=��w�R���ؽ5!R��%��}<��=��=3�=��k=s\6������uD�w�W��$�ϻ=k�#=r�=�pj=)]�=�Y��K�<�v�=�==vf=O �=�n�:�=Ғ=�`��/l��s.<%�C<�ܽ�-=[�ܽٯ�!�_#>8��=Q#]���\>TP��z�U=S����-P�D1)>�3<m�А��@�=:��<��j���,����'!�=�)����q�N3��δ � ⱹsu��6�׽�mG�\�����lQ�?4�#�_�@�)����=��0>G'�^IA>,�n�0��S����o�^L==�P��"�U�y�X�t<���F�<��l��� =~f]���=���=ag���E=�B�8F�=�m	>�K�Z�L���e�۠�=�@	�����W��<G>7���<�%���=�϶��U-=k㪽�e�=^>xC��<O�2��5�ս��>��<�ʹ<��T=P
u��H>(:��l<R�Խ]맽ˢ�=�q�=$��<�X�<��<���;@;��l��2��w��h�#��9`C��l�ѻF0�/Ż�@�=x�2���(򕽳�~�ƽ�(b�1ʞ=�P�;�2�=�G�(��=n��}�c=��}��,Ƚ��Խ���=���/aݽ�>�좺�J��]BD=��<���=5[�#���9�<M� ��l��㌽)�y=gi	��U�Q4�螯��/����;��.��@<�DY=%�������4w���3����=�Q�=]=�E�=���<� �yG�K���GR�yc=#��M�"<G�!�{zT=�@=�s���P4=,��<�M3=q�6�)�Փ|=rқ=��W�0��=F50����=V�Z�'�]� ȇ<����x<����dҺ���=��J<춻���^<���<�Ѯ;M��<��N��h�<�����h<i�=S<.��11�	�ʽ��<rʻ���Q�n��Z<<mZ�pI$����=�����,=����]<J.
�qI=A�P=�p;k���m����r���'=1%�=A;뼩1�=�t=�=|%�9ٳ���t:y��=i�=8��<����9��Q��=C�[9��&�s����7<�_�J��=��e;�=={�=���<�k��V��:U�=���=�}��;������K=�{�=���<���=9ҽi�� ��_�����=�"=�Tc��u�Я��Tn����<��#=oǽ���<&vI<d[=��?��r���<,t:{W�=qg=tJ���C��*W��/��<�y�=R��<��;Ix��أg�h���Dޠ=wg����<���kp�;��<#2��K>�,t=�]:I=6':���=���=-�����e�&�}�׻3�B�j�,=B9�=IT�=����k�=�#\=���l�"��e��l�M�=���;���W�<�	=��;��=x�	�	n�<�"���<'���H=������˽*D>�=�u��7�=��a�G>b�<J?��T��m<e��;��,��F�񷋽)؟���S9�=��Ἄ0���*<z��=�;S��%|�r@�6z��]7���=8
�=��=!'�ie>Q�=�\^<��~�Fʭ�.Y=|��<�����<>{��=�=ɑ��{�=��W=Qz�*03<�y<�r�<��=�����m��f�f�-��<o������� �=�p^=�Y�̨�o�K;W��<���;��=/��=��н�#Z=��:=��6=�r|;E�����=·�=5j�S�I=TE==�{�+�:L�Ѽ`�=��<����*�O��-"��'���,�;����᲻d��h*���tE=�3��QP�y�����S׽��,�̼>kĽ��=M��=����<�
��D���B��\��,�����<����;���!+*���o<��=Տ~=�֬=ĞZ������ <�����;��=�L��估S�;�,��׼�x�=b[	������B�=�B�; �<�l;�˴<�i =?x�=V�2<���<�,H�,B���>zn���a&�5 潴���@
��8=�w�<:��<���=B����	��@R�N�Q>H>d+$=R��,��<��<ط%=-a=m�<v��̓=0%=�F=F@;c<�<�s=��u=��&=�?=|��<T�=燣����\�*��� �ә|�{p�=r����= 5+�1����d%	��[���E�<�=c��=�<zM��(H��k���"ν]�<����=������=�P�=�;��/;=[�<�K�ϴ�<e�t=��d���p~�i������p�<��=U3���=Gϖ���R=|��=T7�Bz'���=N��.,v=ɡ���<�o=��`<�%�<z��<��ν�M�<�.D=4�#</d%;sA �:�<�jj<�挼�0�=TY���0��Q�=�����g�<M���e��;��s=!�>UB����t���Nd�=Xʢ��XĻ|z|�y�����<�� ��Ҵ<	%i���⪼N�0��=A3�d'�󉦻�����u�=��f���=�L���Ƽl�4<Iv�=��13�<��9<�G������T�=��=F �� q=a=Η'�F~ĽF8 >�o��Aź��=/�.�=3���|n���<�Ϊ�!ta��
���U�2AE�韇=��<��ǽzR>��;�/;��v�=F�K�@=�R�=�6�=�Ҽ�����;���I���<߻�[�=e!<��߻1+�>B��܅Z<���ig�=�':��g1��A��`��=�Px�2��=Wq2=`��=磕����==]���8�=�9�=�J�襢=��Y>[˛=QR��.<�<�g�>�8Z�����C�����j�=GG�Z`s=����2j�Qs�=c��쉽ڐ�=��=�=B�5��؁�V�����	�q�N=���=Kj]��+�=jk�=�Vh�s��=`��>2�-�Pʚ�9�!���=F
���P>z@>uI�pY>x38<jP>=�<���{�=��=$g�=c��q�ռ�<E>fǱ=]�->�y�=/��=�
�=��$=��5<��=��<3��-��=�
\����=7?�=+[���A=|��=2+Ƚ���=Q�ǽ�I=��4b�=�>���<�>!>S����@<��wr��>�m=u�=�q=�,���~��d@�=����.r���&U=���w�=�I�<jh�<�Y�=�l���a=��=�qϼ$)=8VQ=^�.��*<7e~����3,X<��<�ʹ;�1P�U �j8��Tr�z�{=�� =�+��`����$=��P��6>>��=I �=���=d�<{=}��=]����ῼ\J%�!+S<��=��<��<=Ge<w5>����� �=�=Yr��CM=-�)>l׽<۰#=%f�'�=�U>��=_�ֺ��)=XS%=�:�<.`=r��=��=�,�ּ�Q8�od=�L���u��04=P5<"�N<���=��(�Lw{<�N =,]��u+�=܆� �ۼ:@��(��P�]=.dt=��=��!=y,�=�)<��\�Us���ݼ�=��%����|m�<�&=\\=sS ����<��=U�����������������<���=��j=����>��K��p�t�t>=)�D���=����?a<*��=�ר=\��9�=rn�=."�������<" ��7��=���*d��@=�.�;\��=S�:���<5��0��w*<�Z2�_��=�Y��O>�G=���<e�?����=����-<������Ͻf�=�UM>��`=t_��U��FC�TkP=� �@��<�����8�f��'�=�2ڽ �ټ�[�=^ ݼl�[����*-:>�7��R�>���<d6�	�>�I���s>2ż��V>¶Ļ��<܌>9b�=c#F=�D=}:k��㴽��7=�Qѽ|C=N��<O�j�?��<�?�=%�༱Y�<Aq�=��<���=���8z�<WE���=���=�4��7��=ɕ=�$�}}L=5A�=�6=��g<� =޼d=�4��1�=� ���L�=���;�=$D��ڬ�=���=�������=E2>E���M���g����L>�8�;n">g䵽r�4=��^=:���"���}� �<�F�;�4l<�ח��v��J�����=;�<�Tw<hz��=A�>z���և2=�Zt��W�_>�h��G�>Th=2��= Y�;�k����>,��4Ӧ���,�^�=�k�e��=�=��e>�*=nd��]�=���<�47=���.���'H�=D�=/�N<��=X�>Dz�-�=�=�7�=��=n
��O�>���=�;�y��F�3���=���=7M�=a�J<���<�^s=��
>=⧼��=wZ�=���m�=�-�=��<=]=%��;�B>R�<!:�=f�;=6���<�Ge=��-���9�/�<�M�f��=�oջ�>RS�=wL��FS��ҙ=��=�,=���=��l���9:7���QE�=>�D#�f4=�T�=�W���p��I�<�5O>x��=�&>|YC<�dI�L#
�w���\)��RR=|A߼|�~�BT�=����#C=�A��DŽ(~GG����"Iｻe:>=jy<��=Ć^�ڀ�=Ec>����Mˢ=��=�!>R/��8��<��>\S�VG�=�̻��߼�g�<&��;b�򽂳�=��;=����>�W.h=�� = Cv<��;Cը�M ,=`1>2G
EStatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp�
EStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@@*
dtype0*��
value��B��@@*��C���wX.�o�j����4�=�A���,�|g�;�+d<}D9�H��<G����z`�h	c=Pռ�`&���6�ol�;a�'�H��<J}9;�T';=�5����=su><Mn�� ��<�7=���?��<	��=�~8;9��=��=�!.<PP�=�I�=ϧ�;3��<
B�<{;�=��<=��=�=;a�=$)�<���j�K=�A=�>>��<����5�=�3�Xct;Zu�%������@��8�����<�q=��N�='%�=�"=N��
�ʽ�F��X��9���=�e�����<B=�J<���M���F�|(�<J�N=bl=��=d��=�Z=�C-��dV<g4�m���·=�b�=89��m+F=�dP<$�=J�=9g��
��w��Ow�<�A�<NI����=Օ<�1�=�>6�=eIB=�<`��M�=�e<=���=�&ս÷;��=��<�@���� =%�$=�7$=.J�"Z<����`�����=������=D�(<��9=���<5
=��4�꩗=��G����=d�S=��>P<�=���Ji�<)Ͼ�U��ޓ>��_E��MȺ<ᯜ�bul<,��=��<.�.�4f�=cV&�r�>Y�л�G>6��<���-��<.��=0@�>�(P;`Q�7�<�B�'�=ٹ�\I>_�t����<���<�a��������E�u��;����U=�ѽ��=M9��>Ǖ=�~���4���=Ṻ=k"���a��n�;���<@����H$=�c��4Y�S ��sr<FU�;Le5=Wt��\�=��Y=��o=X�;P';T�4;A�$4=U����8�<��<�/h���x�.�����"=y�;M�S=x`ͻCǥ=B�=��=��5=�-�;�/=5����̾�\s=D��<��<)��<P=0:L�<=��=�g,={=μYjo��ض���Z�o a�?�ٺ�R��M� ͑��9s�<FB��8�<�{<��f=#�Q��=z�V=lI���%�< ���|=b�=$w���5̽7��=$/<�7U=�.���*=NI�򞲼�f���r޽3�=D���
=>ŕ�h/�=���=�q�=��&�xC��6)��<}>��;�7��Ye��}i�GT�w��g,��Ֆ�Y�=�ч����*>�{	�f�н>wּ���d���+Ƚ��ѽ�8Y=O���R��<p`�=����UG���=�䢕;h��;��9=g������=��W�~7�rB�=�=�A�=�MϽ�c�=��n�I g�Huf=p=���{<c��<p��?t=-4=eh�<Yf�+V�<���;Y�=�]i=ڜ����=a��Z0ɽp<D�=@�==~�=6�=g@&=?�%{=�߶=�X=Kf<�V�=^3�=@B4=�4;r�m=�4c<��Ͻx��-j=9h=��;K� �:ܣ=�߁=iZ>a7�y�7�{�����ƴ�=VF:=�J�=jrѻ��=�*=��<&ON<Br=�bU=��<6���o��J}2=^x���(=��=ZD>��=W7=���I ~��X��z�<	�ν�t��?P�����=5��HսF|O=���Wpz=.�л�м��&>Vb���N@��%J&�E>���;;a��v<��ǽ�%��������䚼W�?=�_����=Di	>��<��g�1�<�$:��s�H�p6
�?�=&��=�9=h{>Nm�#�S�����#'<4\��3B���=��h=G���경��>��>Z=���<��8=��ͽ������W=N����<[Ԧ���<�-�id�U�F=
i�{�;�t[<�'�)��T+=�2�<���<ǉ���躽_޻��|��ё�3eF=�����w.�O��=���<��ǻp��-1�<B���1=������%<�%���<�ւ�����o�=�uܻ# =z�=|�=���������Z����2�E�G5G=@�<q<�7�=��5���{=S;Ew<��9>����
7=��	�=*��C�=RL;�^�<w[S=�-�;J�U;^�=�0�"<(=�L � ͌<<���=ҩ8=JFO�G"�<Q{�<o�<X��-;�����=��)��ת<�=�=��<�5=�+��$$�<C'd=`Һ<X�=�1>�M=<�>G�{�
�
;�)@=�ҽ��=��y=r��<�e�<��s	1=�$L<��>x7��\�;�,<��j��b}�=���<P1>�q�=�3=z8q=�pH�e��=�j�=|2*�Q��=��N�W��;$GV<jV*�HY=w�(=|�'��F�=�z�=��=��<}�=�ꮽ1|�=;Ǹ�ui}=�t?��=���<�h=ӹ�=&���j�q=|4;ר�=T��<�̋��p�=x�=?pV��p=�T=
_�=.�N=Y��g�O=�;=!(�=bs=��[<5f��z�m�[g =�'V=��Ž�N[���=�6=egc<쉇=�Do��[M=$��Z�R�<|Ͽ=�~,>=�F|�����C�����=!V���
����<@��#]&=�:,��}0��ߔ=�(N�+��<�&=0g=B.�<��<[�-=`=<�<�-=�w==��
<Jp{;�Q=3� �I��`�=ˈս�Un=��=1f?=��μ����=�=!�[=A)���h���r0=ޖ=OVR<A��<� �s�"= 9[�YeQ=e�:��>&�ʺ�ݪ=c��=ĺ�;����}z��h�=%倽��=�o�=qɫ=f�">\�~=�x>����<P���i=��$=xӻ���'�<���=�U���=�P�=�خ�JP�<4+-��M�=� c<�!�=O�<�*�=È���6,>2nF>=�=K\�=;���>�����_���>z'��>=*h">ra�;��[=��!>��=U�> �>$��=�>�M>�>��7��C�H�K<� ��^�>06)>l��=Kν��_=��]>�h��{$>7�$��8>�Q�<qLU�&Z$>Q�
>u�9>'�>��F=�^.��x<�0��L%>$I�=���g-���b_=T
>�+&>�K=��=�V���=�
%<M��<	[�;sN<qWn��2��;ݽI��E񹼚'>��/>��:=*P<��*�iNy<n4��%���(������<����8=�6�>�OԼ䴟=����H
� �h�3$z=�ѓ=�w*=^8���;eD��O>'8M=��=��=/�j�7p�+P&;%��>��y��v��gnP����=�V >S�=#��"5���v����h�S�A���-=�A>5��<�νI���~z�z<�=~��>�8�b�H�+��U9��My����=��8��I���f���� =3��<�`Z<�柽��ͺ���J�^�V����$<�k�;��j<Z2����6��<���@��<z!���9�����"Z<3�=�!�<��ں��<ᕸ�W��:��%�� �=pK�<�Z<_�^�=�R ��ʽ���������<2�<��Q<;��<zi�=�Ҫ�(
����"=N���zT=��;�L�gԼY�8�
��=��N:
��<� ��T�'*�<W'��}=����Up�����|��=��u=��=�$��Bx���<�=��<U������~�<h�U=a��=��&<0=Z�<�a��l���	$=�y='��<VO�굚<�J�=9@�<��>�f<qa!��<�����u��B=v�#<��}=DEP=�WG:��>r)u=����齺L�.�b��=��ļ�>=y�(=�b�=v�S=c�;��<�d=7L]�F�<E����4�=�wD�yd[=��(=%b�=��<_�%�S:�=f�;Ӣ�=��Ľa�>�@=���=�Q<��=xb�:��=��=P�c�T[�]A������=x=ϻ�'t�=����\>�j�=`�x��B=���=?�=�k�=~��=�OD�)��<K����>=�]C=���=�l���]��'�<�x6>��=k^��<�T������Z��\�=�ϵ=�/=�9���<U8ʽ��!<b��T>�=��=)AF��d�;��/�C�=�`#��M<�c<Ȅཀ��=�r��ip>@I=�*�=�87�������	Ѡ=�b=��=�=�=�>�v��9L�NJN=���71ܽ��,<�ߡ=,��=���<]�!�z^G<�J�<���=�=��=�k�=	9�=�ψ=�� L�����%�";�c�;��=�䢼�p��-��;���=��>]믽��=t��T��K�A>���=������>����D'��=�l=�2�=��ӽ"���{�=:j��NX��j��=�<z�8=X(=��<��,=���={i<�H�<�W=�1�{�T=2��<6�=���a��H=��ݽOսpl�='��:��>Yh�=���=S0>�T��?�Y=�'�=+lݼW�7=��<��#>��[�j�+=�d�	�T�V㌼�Jj��j5=�$l=�FĽ�=<K���h�=�.�=8���2�=�#�p�н��=|��<c�ϼAX���F=3콕�#��b:�W��=���=]��=�����=�/�=��h��Ò��z���載G
>p�ݽc���C���^�u��rżH��ߜҼ�亽��S�w�����q�t�r=��4� *	>\)��q��nl\=�M��=Ub�;���#��=>�ռ	���xת�MV�<	�L< 賽�X��O���v��=�ɍ<�rg�p�>��6� ��Y8�=䠇���V�X<5��w�=$7-=���q >n��������$�Q��<�o��nS<"����ф=�5Ľ�d;=��$=�C>`5�=�>�;yo�=�Zǽ�o���׼�  ���U=CD=B!=0��=	E����=�y��H>B�=�d�=Q�>��=�4�>@$;>gr���P>�cȼe�G�v�9>�Ǽ���= ���:>� _>��3>��<�'=>�r�=n��>��=g͑��u����ؼ�#��ҕ�=.�H=^L�;�ĽQv9<8��>7=D'>s�<4$[=���=ǭR=>'g��MP=����3=��<��<�c�<i�>�'�_c�=M�E=���=B��="�[<�c=}-�=l<�=�h�=�烼X9�=�1E�Ԏ�<��5�R�=��ۼ��$=�H�<M��<�,=�ڝ���Rp<��I=��=P�Ͻ��{=Nk���˽=�KR����s�6�t˭=,�Z�]�<�d�<��=��=v	W=O/�\����	����e3=zH�=�Y�=�*K=�=B�F>u�=����=���=}ֆ<&-�=I�=z�<>/�켬�<[�� 4b;ᮟ�B�|<�5�=Seֽ�3ҽ�ۛ=��>�a���ɻ�T
>�HW�Dm�=�c���=��S�������	���=����	��E���=�;v���ă��ԅ=68���"�E��<�O�<l��=�q=ͅ���#�=��ü��
��{T<g�<P�S���= H=R��<:�g=���?�<:N�ʎ>���������<޹V�N�==���A">2�Y���w)='Bǽ�.J=o"�=��=Z�=[�y�U�7=𭜼���=%%�=�l<e# >�>��*.��F���碽�"�=�L�=��B�Y@�=ƥ�h�<�<��=��<�2�?ı<��f�k�&���<��={���_L��x�&=1!.=����'��=X�>b��=o�X�<�|=g�=��j��=W��(�=�I]<:!	<�~��M�nS,�ڼ�;��=ʋn=s2E=k�<g���/K�R�e<K_�;I�=���:P��a0�Bh)�SB��U7�<b�N�(�r;�I5=L8�IT=�!*������@��b'=R�t=Ԋ�=o:�;�ڼ��!<_֣=��r��"s=RK�=,!���W�=r'�<���=�ɱ��H=];<�C=N�E=`��/�S��c����=$N޼�Y�<m\f<�P[��s'>!욼n�=���=��2=��)>�k���%�=<1=!/�=5�¼��Y:�@$�@S�=o�=F, >�v%��岽^Ϸ=9Hj=5��=��$�p=y�(=����>뎐=�p\����=��<�j��	m=��9=-w�=�,�<�ŉ��w���懽��>r�Ľ�W�=�9^xU�k�C=�� <Q��N*=�m���ż�$q�Q
��p�<zwս���=��Ӽx��� o�L^�=�7�<�= �]���c�< q�2d=tٜ�� g�$�����ܺ��<�8�Ґ�;�!?=;z<�z�bw�<2!0��E��ѱ��I�G=
>��ぽ?�ý[B�<�R|=V��=�<RxX��㼅>h:õI=�F;h��th�=&��=X�u��J�==Yb�V#�r�b;M^�<�L��k��=�* =�`�=ݴ�e���3O�=V4>ӫ����R=�b����<�4l<�{=rn�<ID�=vS<Ҝ3=۫�=�h1���.<%1��|�Ѽ6w�<#�<�b<��<�}&�k�3<$�<`n��^�=!����=BC�=��^=wo=^@5�&�=���1�n<�1=�%���,�<%�<���L�=�WR���4����;jk�%�
�9�=����[O<쑨= e�����sI����=��=��=I@|��gԼ�|=��I���=C|�<#��<��<�f�D��=�L*��a�=��V;	v�<xH:=���=9܃=�:�=]VV=�1:=]A <����)j<�8�<����<�=� 2�ͼ>o~>�9ѻ?�x�uR�=�-�=��<<���=o�=uNO=�\�=~�=D�����*>~���*i�=Q��<P��<�UI�"�=��>�?
��R>�I=����==!�u=�iL�<�;�KB<+q�m��;�H��B�>�>�=>�����`t<'A�=�w�<��U=)�=�e�P�9=;{~�Ӷ<PO>�0�=t-����_=�"���x�ٿ�<kx�<�Ի~�|�(��<C2μ�i=����2�e���Zhu�Ù%<��?<��o��Լ�=�87��|��rMȽ�D�<Y-=�FǼV;��tG,���<%�ռDx<���;�v�=��?�H]<��������~:\��< ��5[���D=�I�:Z��+��_w��b6� 6=x��BWG�PO}�֚_��蚽?�-��$=�+#=���6a��R�<Ƞ=�a��z�N=w��<_��ఞ��N��F39]L=_3$��K�<�F�<��;�~��iF�;�p����N�>�b��<�8��,C=G�=ɑ7=�Z��#k}<�z��-���%q�<�Z�<��<d���j}� �S=/<�8�=�����T=c~e���<8S��c�<��<�=���ѵ<���S����=����@8y=(��������8�;~���Q<'�:���\��:���)�I�3<C��j���8P<�N�yI���O�;�H���<�X=��v<��=G�<_�=�� �=4=��)=m)��?�=\rF���ٽ���<���{.z=�(3=*��=���=��>L�(<��E=U�=�˫=vO=���=t:";������t�=r��0�>`b�?�,=|8=cq��هI=e�4��Ȟ<��V�TS������g뼌�>���}�<��<"?<��0�G��l�=���=k�=()���V������S���>!��<�m��y��Xa�]����k�=Uꂽ�k>x���=g��<Ѿ��"=��>%��=c)8��>�<�=E}Ӽ���=�И��
��q_����.�=����k~<"�y���<Δb�F �<�^=d�=���4⼌׃�>La<���=�5�3>ͼU7�Z*���=ɒ�>�"�=�����ͽ�^�<��C>��D>U&�d� �Z��m�<���2@���>a0>;�Ƽ����Y=Ɍn���佬Sx�rIڽߵ{��N2�7
�<��=���=U@�<��!=I,$=��=UE=b�=��c=�$�<dp=�Y;=��O=.�j�4�<�q%�N��P�Y=�w�=��޼0!��DK�<w�=NR<�n;�[8=�=4�,=�O�=}]���<�N~�ƒ���1�<�  �PU`�: =�S�=&�=�1z�S��=I1���b�<��b��l(;ꀕ;���:��B��+���w��I�=d�#=v?y���=��h�J��=<�=�eܼr�����z� =�!�=ijg=�_��p=y�D<tF���͵=q�f���x<�-<#�>aK�%$��U=�'��7>�2�1h=�h>��Ž�߼�\#��Z�}�=�|Z�憵�Sa4���	�� ]<�6�<��7̀н�ݘ=ʱ�S�=�P�=r�)=�E���ý��=)�D=��<S�ս��Q>�:���>�W>�K3��2��ay�<tn�=��=�,�=�t�����=���ij��5T����=T�
>�e�<�,:BL�=�@
�1�='�f�i7�;	ż��Z;b�=8(ɽ�o#;F�3��:�=�բ=#��<m�;=aȽ/=v���s<�"v=�$���=w�:rE�<�n=������=�=*���Ӑ	>m����N�=�+=¸�=����}3����&齎NU�0,��U`<1��)$>��~<j��=-/U�K̈́=�Ȉ���'���>��<&_=D����=36
�O�(��;4]�=�U�=�1:�c����2%=y�=4}���ي=I2�=�8���=5oZ���?<n��;���t�<��R>�<���:~��DAN�s��=��,�'>&=���Zm�<a�q��A��	=�T=���;��n�k=��#����=�cɽ�,�<=��<�(=#�=T{�<�'6�{]=6a�R9Y=G��<�/�=�j\��=�K#<���<�K�=o$��Ti=x����[;=$�+��Ȅ�(����4��<`=>�;��;v�<��S��2�:t�"���pTɺ�V<�j�=T_=��<Q���-=�Uw=$	M=��ȼ��=�0d�M/:=;�@�aC�=&(�=c�<��=n7<�`�=%@���Բ;y#���jR����=��</bM�S��=�5>	��D
����=`l=F�0���M��G�=�ŭ=:n��D�FI���L>�t�<N�=03�<I�P=�=.=���=b.>Ӯ޽ �<*ò���O=\a�������ốN=z��<��E��r�;M����9H=;���>�;o���?/=Zo�<�� >��>,�|=�༴��<tp�5D�<<k=A�@=��=u��p+<�o>�D�=��=�P>�,��e'����=�!=�>��>rS�<��>�/����=v��L>3�=v>e�}�A+>��H=��E=Ͼ^��k$>�H�������G�!>oYI=�#h��K�<9K�=oWn=�J��x�>�|�=o�c=�A���~����8�l��=4*=�P$=+�K=�PX���ڽ5H�<�:����O=eB<��M>���;B,����k6�=�>K~��\�~��=!'	<8<[<�Ͻ��4>n�<R�=@��]2=�1=�ɛ:�HO=�]?���ٽ6�<=��޽�y�=��=g�=�i�<�HϽ�<= P�:��<��,=��i={L�=C@�=���<��ݽ��n=W�轕4�)�)>�#%>�ټPq;E��7W|�=(�->1�����<=��=�ȽE6�&�=c�P=-fƻ�>Hm��������<��=� ?=Ş��Kge��ɍ<@r= 5Ƚ�z<���=����>�9�דx�h��=�4�O�<�����7>Y�L����;��|=Ee�=��<�t�|��=���1~=A=�P4�c��=Ɨ:Ƿ��.M�x���ܚ=�k=�ל=|��=m��=M�q=ٗ=SZ:�$ν�>�<�I�z�J�=�=2�|<��=�]����=(>���=��=ą=ȹ���[���<@�����^��y4;擼zآ�cA6=i3?<9k+>$�;>��c=�଻��<�2�<�Mڽ��$=��g�A�t����=)�;�n<��=�U�;ݰ�<��P�	&J�|��<C�=�`����<b�:�Og�<8�=�y�=��.<ɔ����A���\=U�N=��=!�K�r�<C�Y<]�'<�-�=��ȼ��/��5�=p7<=r��D���cֽ��d��Zx=xN�5tt�d"=�Z��&U�=޳=��>,���;R�h,n=�r�=��:��;��<=��8=,��^�F�Z���'�<
�뼯�>��n=@К<��x��7�=8�/=���v�<�(�:X�<O;ȼ9<<[ǿ<��r=7ϑ�x���� =�.�;D,��(�=�C���H�x��d=���=�s��Jp�4/��d¼"w�;��� �<��y=��h��sO=8�8=��b=�
@����1�<&�=�e=T*�BRU�}$�K~=>��<�E�<����$����u;F‽뤢�M
t=�rM������.l<yȝ�=i�<_ۈ���U��+���[�=l���d;�W#�RJ���%�=�8.=u����޽�����r�<���!�����={�<ZT=ڣ6=<9�= ʉ=g�=r�$>(vv��e%=�8g�и��6�U=�eｒ�2=�q�f��=˿=�t��)�=3�J=(�:|g-=oܼ K�=��=l��<!�����<S����-~�ٿ���=}*9���&=�UJ=evj=/A>��<X�C��F���;���QE=q�=I��<3�9���=�Z�����4��]�">��*<|I�����bߐ=�<��m���-=�x�=$_����=&�w=VE�<�ɻ��~=P#>3�=�W�8�A=����F<�[����1�!HӼ���j�9
>h¼A�|�N�=�8�8Ͷ<�ʑ�I��=f{�=�\�=J`e=�Vj=E�<uW�<T�)<p��yFf������Ì>��<y�%;*@=`8���ph=/�W<���=Z߉�X	���"�<R�*��#�=�$%=�W�=�
9=sͦ=i8��#J}�8u�=D�>���=����gϽ�3�=գN=ZMA���=Լ�=q:]�b�`nH�S�<�t��NໍA2�<->�� ��k�=i'�<�W�=g�Z=k����]'=�l���� �<��b���>�C=��:=9 �1��c�=�#�=��l=���=�<�P�=b��Z�>�=i+��[	>Wi��f�;2�, Ҽ4Ay=b������= ��<���E<�<,L�h�K==�&b<vC���Km=S�=��k��k�����=I�`=�����Bk�ގ���(���%T��ˎ<[}{�Z�<-`���%��V�ę�;E�<�*�<�G�<���=6ǵ=���=0�=^�=/� �}F��S�s�����	=f�d��=21�=}T�=Q��<�v�=�=�x\=��~<ɑ�=L'>�M	>��D;l[�=t&�}�<�0<�Q}>%_=�?�=J�=$�}�<`�=�:�=��9>4����=�˾=+ 轖�=��׼�[�<�vD=�<׌�$;��܂�cޞ=�>�h<�`M <	Q=f�s=���>�< $�=.��R�>�"
��.��J8�:h1�=�ǽ���;��%���>��̥p�3�>\q8=FY>t�><�>����=�]�=u��=�D(=㪺��g<��m��E6>�+>x�>�yC�c̪=��>�$>7 >�v����=~���Kq��>�I���k����=t,>#�p<y[�=���=�<��亿C=H�ٽ?M���zT�p�p��S�=��ļ.n=��=��������� �u�RL>$G�==Ƽ�= T�=��>�8�:��;ү|<��=�)�=�#����=_�<#h<��Y=-��<d�9M=T���<x�$=EvN�_@���F�<�6˼��/��&��y/=g�<��/=#��=���9c�-=_'=���<�<���="���W��;d�;z����m��%b �?L��Ԫ��\t��]=Hܐ<������ŹSE�R���t�2<�0o=/\�¯x�
���=�Q"=Hel��Ռ=%�=��� =���<Hh�<�^<=�*j��V,��כ���_�"�#�Xr=�ػw�=�F𼚢��$1����k>�`>�Et>h>�O�<��B>�B>���;.>{��<��'����>�Ѱ��i�>j5T>�}>ېe;C�=f��=�OP>6;�=�S ��Zx��&;���`�=��=m��<����_�=��=��;��=<L�;�a=��#=$�=�>�<)�6=/.��O'�=�"=����Wp�<�ؘ=�ֽ\��=7=��=>��%=~C�=��;=�m> ��;@
�˙8�vh	<��G�a+�=���<=�V��8y�{Ѥ���E=���=��i���Ƚ���<�ܨ<��`���R���0;J�<*2��8���׎=f�=�Ѧ<!�(<�=��K�<D ���9�<��==Lνǿ�=$=�y��3�<Q��*�-��}�����<�A�<k%�=|�et_<�Zc����;N�Z=�6�;���܈*=KF�X��<ݖ�=�<u|<D઼��N<�g�%㫽�h;�*7�<19��@4�֋h<��w=n*罕�P<A^=@��<�9�#����u��`���s��v�=Cɽ%jT�Z@"��P�=�����2�z�]�:������r=��=no�P#��������꼺E�<M��=�8� yh�f�=HNX���*<��">���ͻ�����;;��<�8���Ќ=����p�r��s��:�=��A=�|~=�z<<=ʽ3�߽���<���=�d�E�[<�/�=��d<�Z��?F�W�W��3>�A1<��=�};�ko���ϼ��
=\���V�=��< ��=��2���<q�����e=5������>�������`�7yH�����v\�<�X�=lB>z�?��R�=�@ؼ�nս��>r �=C�'>q�=��==�ѽ��D��`����ʻ07�=��=�}3�?'=<�=&�(>�@U>��+�x�<�M=1�����=cR�=�k<,m�=�'<Ѽ=��У<8�=�Z�=5�<{����N`k<�)=���Ƿ;��>�'�<��X=.*�Z\�<lF'��%h=����P��=�;$�b/�--5��y=d�=�D<�S=�g%�R��=��<��ؽ��o=��==6x�<��>�a>�1,ʺ�}�<;N�=y�!>��@����=4�>" f=�x�"�C�e�a����>�5<���=13U��4��N�=�0�=��V=�Q'�)z�=}A��r��=�G�={�Y=Y�'=���<���}M��P�@=1�=��Q=&Oӽ�2�<K�����;���;���<�ɳ<i�qm=�#x����67=��r�j����`<�k�<hQ_�K�~��J��Ż����<��A���>�=�ē�7�<�,�$K��![�U╽Q_>��<J�d�eE�=�����U<s�M����<$Ͻ�w
=�ü�^1��\>P�>�b�O�ؼ��	�S�������<��7�</TG;5#��/U�=� �t�;�/����� �@��=C(Z��]==�:Ž���&�=E�<�2�<�A1=f#>|*��b���]T=<�M�=N�T�\=i�=���<�j��I�/=��i�dDW�ه�=�2=�%�:�q�<0�;����M��X!�=0�D��7���3�3�#�a���@�T��ܿ��26�-/�	�<���<#@d=����\,�����KV��Ļ����!�"�Ib���;��|��/;��i=M�D=_���u�+��n���F�=Vc�=�ݺ��������j�<z+��[�<i�>Dz��Ż >�  = ۝�yb��*�(�b{轲Q�=�p���!=9kɽb�=��I;�@c=�26��e�=x%��2Z=���<1>�ԏ=q诺�<i�½a~0;m�=a[�u��<���=܄v<�Z>�Z>�`�;Q��<V&=e��=OU�=��=t�=i�=""�<=:M�,>���e�AqI�0��=�pܼ�A��s�vkF=�'�=�<@;<�=���<�tL�m��=��=�q=����ޞ=�M�<}�⼔*����=H\�<xR����\���?=U$�=�l��O�cm�=�Í�+>n+����=��[=���=Z
���d�=�ͫ�R�y=3����=Q�1�j�u=>�a=��Y��K7��<�=��=�<�e���y<���<��K��V�<0��=(�*<z��<8E�=xE�<+�[=eO=�<p� <'�n<�/��l��zp�=��<���x<�z=l!�=\��=٪���{O�.��=��|�p#�<Tl�=).v<$Y6�!
m�aǻ�Cּ��x�"�=�=�F�]!��'�<B>Z=pV��ӽ0���z{A�%�<��I=OL�A\{=hZ[=�:�]Տ=F����=���=%�=P�:�6�<�����3+���=�m8;���j
>�7��g��ߨ�<Դ=�?=���=Sf`=��=��;��f=_�=B{=u� �S=��P���=�Sy�	��=������ν��*=��=��=]�佫�ҼO͚�e�ʽ8���"<>YM�=�YP<u�?�����\<o=<A*W>�1�=Y>V=�XսQ����kH=��=3���|o=\R˽�c�<9	ڼtD�����;>n�q����ﯽ�fȼȑ��FwӼ�j��K��J�&<jDh<x��=Y"u=;K˽ӵ�=Zfo�x���罽`� ����=���koʽ}�
=�����s�Vf�<In�v:�<�{�<S3� Ώ=�9�=�O���5#��I$��䶼�*׽��׽��p�r^=*UR���D</�>�SB�`%�K4L��v=Q0Y�D�=��L=(������:Hƽ�<�=�#�<��=������b>�H�<5�a�]���ْܽb�B�=I�4������e�5v=|�5�Q�3>|�Q=W�=�=�8_�A�U=�]%�����e;՜��=�<��=k%j����=(K�U����x���U=��=1�S=���=�
x<�b�<¬w����Sw��9�{�+͙;.���7��=��;��<=9���}��=Oyͽ�����:=�(M���=���=(nU=�~�=��<�]��̲��O�>�=%�|< `a���.��c!<�o ��"�U�=� ڼ�_=�+G<w�=���<i�'= ��@�=/g=Vg3�D*�<���=�����h<xt3�@x��·�= v�=���w=S��;K)�=�e�=b"�=)�=�B�=^׼U��=�=Kb�)��=O�ݺ:�G͌<����C>*ש=u)h=�5����<>�;�e=">>����4�0< ��<?�;��޳=���=��J=O[�<$��<c�Z�џ�R��<��=a$=�E��c���?<]Q�=���=�qE���=�[���=1���R>�9��
��=q�f�e�I=3��<�b=�0 �� �=x��=*����z=*�u���,����=�p�<�t<v�=Lzo�ZQ=����fb�<��1=c*����=�$=���=�B
=��,�N*8�ω=�7�<+i���x=��=��a=�#=2O���=?">k+����<��=��T�F�(t�=�No=��'=8�<���=��-�����C �<O�=s��=�ʪ�|t����=��2�;Kk="ak<��%�=N	ԼB��=˗���xx=��2;z��=x4׼���=�Һ�̻=SF<B�f�L�< ����=;�)==H�����=O�W�TrP=�k�<c���:�`^���=��<�a<�[=��>=��i=|�s������nd=��=���=�2a��=8<��=Ij��5>�{���š=p�=r�?����<�$>��ݼf�<c>�Pֽu�=���s=�1>�,�B,��d���Ư<;{v�hg���6=���<t7�<7�==�_<".y<W�-���	�����p>�������Vս5�k�m�<�͆��3-=x�μ��@���Ѽ3'c��6&=c�p�'�O�]+߼J�ν"=Mz��A���@`�p��<�W<�ۮ�C<j	b��^ɽ��-��>O�(=[��=�ؼ	W#=�^�=p�c=U<������C�*=We�ZW&=�^�;�2���K �7��^�3��ٰ�h�３0�;�5��%{I������(�<eg��"�m�J=����6[�<C=_T<�N�;h�=c�<�Alk=;.<;�e<�<���N�=�3�<�'�=b��=R/<���< ���%�.TU=�c���a=S�<%@=Z\D=��<�[7B=pK�=ҁ�<�&F=ǝ=���=H�ǻƍ�<U���[�]�{�r��TG����=�I�<qy�<]��=�d=~�<*�x=,��<�mb<<]<)�F���";���aW�<{.�=��b<��<'%<� ����<6Id�PAֽ���y=A%�=��3�B =�2T=�$>=��U
��"�<����#W@;)�¼�3==�P<S�&=o��<�)��%2����[�=v�.���<�̼�[<��	�ig!=�*�87&���<j��=7�����=W#=@p�k$�����=����<WF*<���F�I<,�B<��;c~n=��
�`�+�˭�<�G�={�=�����O=��4=*��=0�=�ܧ;�(F;�9�=�S���<�[޽���<�I��~.���=���[l<봀����<?��<��0��e=w�<���=�|�i�<�:�<�������Kw<I�ü+Mo=`p�=t�S��<Li�=�@f��=�W(>�2�<�:=�=4+<�H����u�٩�=�x�=,�=��>ˆ=ۂ=Y�����=X�J�k4�=x�ǽ���<�^:Vy����l�I��=�n ��t�=�(�<'=�ǲ���N���:�>=�=Yi�<�|�<���=�e�=O�<�T=1�=�$�;����6Q>9;��ƽA��=dE�=J�}=�x�=U�x��T=����ם=c|�<��D=�[Z<�>N��=��(>�gr>|<�����;�?��U��=<Q�<&����V<<%Ӄ<�����=W�2�<m|=EbD=� l=`?�=���=e�D>��>0����<=��=N��1���iF�=�W�<$��hzJ�t&>�]�<�r>�=����k5�=:ǐ��}+=;5�=4Y�=�'�0�<u`��
7 �����<�=���=��z#�=�9�=��G=��е=����3��=Q@�=���8=��d���4�������������2;B�,=P'U7�?�=�T<���=8<>�u�����=����#�	>li�<4q�<�Y�=xRq��@�=�e�=�	�kaӼ�=�劽Ux�<�(=��ʽ�1W=�%ټ��D=��s;��i<�Z���:݃=��O�[�`���`=�=�%�=�W>#�<�7=x1�=�^�<$ܿ<FV�_O���3B�QS�<l��=���<�ϗ=+�K=[���&>�P�<X��=��=b�d�?e���^��н<�*�a�{[�WY�r��s*ս4�������b=DŽ�q����;�GNc�_J;֓��+����lM���>����*�Z�佒F��a��}��M���ļ�M�={Y�"]�=���=+� ��Ľ��)=	�ýǀ����y��g�<M����o;�6Ž�����
�?������<@�C���ȓ�=\�#=��߼���	��ȅ�=����e��%;��� I� e�������a�����B���<��v�nmm<b�9��[=�~�=�$P��AH�[+A=���=�l�=���=��"=��g��-5�!�<m��=�1=FJ�=���=[c����8=���>��<�z&=8��=
q>=��>=���=�{c�cYM�?����-�=.��P��=hN6=2���:t=p�T=f}��S��d�U<�<J�~=�E������5�4=k=q�EU��R�=�3\:?�Z=�'���<�5=]�˽���;����O��>��Tٍ��伿"߼�>�Y���sּ\<���p�r�۽Jtܽ	���ǽs�S�ˎ�M�<jP<=p�f�=o��+E��=��;�h��d��#�н�D�Mg��R1Խ`�������8�������1F�;.ҝ�
ji��Xe�:=v����귽�¼���<�������F	�ߧ��4���=":��� P;��P��w����<H�Ѽ�n����>����a���/���;��g�C|�6�P�\��%�= ?h��H�ܨ�=�ɼ���<�L�=�dG���B��<=n���#�jЄ=v;=��=���=_�,6��OB�=�)��k;#��%=�(<7�a�
B�<�]6�7�5�P8G<���;;%=w/=�{���F����=���=���<�+6=�̈́��T�= �=��<��9�<oz����P<+�^��җ<� b=�D�®<{����̩�F�=�_�=��5�i�<�9@���ͻ7�ּl�7�X�׼��Z��;u��<O����
��w�=��r�.˙;j�]=u~�=�l=��=�*�<�q�=�
�=�o���`<=�����x1����=��$<#6�=a����=b90>�r��{[�s�t=\��=JU>@މ<uߖ�1=�
��
Q<U
=�ʀ<�j�(�=Q�=L����=��>+t�=� ���ֻ��&�=^*7=�d=}�Z��Kf<�|V=��R=$;k=+Ɔ=[���o���<���<ѕ߼oS�<��&<�K�=_7�=��B=�B=³�=�����(�yv�;Ŀ�<�v���$���\=��$�1v=J`$=&�<��=��Z��=��!>�_�=I���p��xr�=�
'>2m<|�==Z<m�@<ڪ$=B�>ܽ#�t�e=��B=\��<�0=(�ɽ7�=�ә�]�-<L/�=�Lo=cp����=r_�=�z�=��ݼ�U=i�����=��i=�Ȯ�9�:Y^�=��=t!���ۼ��</S����o;�"�=�ֈ��)Լ�:^�5��=���=+X�<#�>���<�]��pw=t���fs<	�:l����S>�X��CQ�=��B=M$�<sc<S�<_��<9��=�ή�+=��_�M���ܝ=�*��0)��b���E�=��<��<�L6;�\>7b�<����nj=G����P���D��f���<~�k<�c�:���=�wv=��?<�#��u6>�1�;h5=�Ҹ=���=��[�ya�=.$�='�=��q���������>���\�<���F=��>(��=���<&��<:�=L��=�mf<ؐ�=X+d>'�'="��>�}»���="�*���=�i�=�N�>%z>�8�f2?>�{r=a�̼N��=\:����= ˤ='�=A�>~����=L�>�i=�.>�,�=�(>��e>B�=w��up���m���ܒ�W�(>���=���=:,�=���=�}>�>C�N=�!:=��+���ؽ��=#>�=�'?=}==6cO=xg=s���=�<`�J>br;=2f+�Q��=o�=h{�=/ZȽ��*=3��=�	f=/	X=+ ���1S>�%<��>{a��L>"+�=�TD>��Q<C��>.~�>R��=B�>�S�=�8P<c�,>�u�y��;�ї=INR<;z>�����>a�B>�=���='O>�o_>�4>O⪻�d��{F�:���<U�-�p��=|�N>�;���;��=%UL>��>a��=��"<O4�=���<^F�>��V>-�Ž�Ȼ9�<��սvֈ�|�C���>�n�=+>����>��<.И=��8�L��=�G�=�J>8[B>t|�8��<9�=]T�d#�=wϳ�c���j`+=�ȼ�������B;<��=�p���eL��v=f5)�;hH<�=hEe�di�����:kB)���e<_i���7���=kq�<J�8"�=G�R<�Y ={�μ^4I�����!=���<S��B��ut�;�џ���>u �:�fo����:ȉ=��¼C�#��Ԕ��s=Z��[Q<]��<�Cɽv��A̼p
=?��;p��<��.=k�,=_>,�+�z���1���o�*����<�~�<誼9 D=�G�<Rl5���=R�9=��/=�<�i�;��u>˳�� r%>ϛ���;L���@<��=�� �*��h�<���;�,�=Z�;��=_�ּ4�^<�|=��>������J�='����j=�3E<�_�=1���/�=�%h���=��=���<�1L=�<A�";�<�b[�>�D�/�<]�f<�����ļ=�8b=�N���L�=a��=8+��B>��
=j?&�&�)>���<���>Z�=�K��h�r=��2�в<�*���=�:l=���=��D�����<����9켓Ih=�н����R5=4�<Vq>r�R��,p=b��=)�>5�K=~X]=�Q�=�R=���=��"��r����%=�ݕ;9���G-(=ߓ2=0�<&ۈ9�� <���=iQ�=d#�h��=Q�t=B�<�9l�� �=���<�| ;`q�=`�=/Q�;�%g�ŭ�;�q����g=�!��`g��'�<���=��e=����0�=�7��m>�Y���=v:o=>p�=Q�V=�=8�<@�����=m�==<�)�)5=����a�	=�2�=�=f��=�N�g��g#�=,��=FE1<�;=�ߣ���=�D=�ۼ��=���<����*L>�>Ƭ��Z�=B]�;6�c�Z�"=&Ƽ��̽���=P �O�(�k�=���<�:#�>�.=D<=g���A=�w�= �=�|�<�c�=t>��>hO�v�U=-�=���;�>�zj���	=��E<�l�=�ta<��Ѽ��=�->��=���<�+̼ �f�"|Իi�0>0*�7S�������z<��1=���= oA=қ�ˡ=(�\;{Un=X�=��ӻ@��=
%=����a�2F<��û#a�=��3��f���l��!<T�>E�>�n9=b1�=/��<#��;�k�<��{=�g|:\p�;��=&�=򿁽.�;�C黅������=n����c�y\�=ۊ<�j=M�Ƽǉb=��&��_>��,��5��fX�:i����@6�p�(�!��<<�Լ�4Ͻ��M�k�g��u�<hϽw��hR�=�$t���*<ȯ�ȴ��ü�<Q�����*=^~	�>�v�qdݽ�Lv���p+���oJm=��D�ļ�>��=޸м�h�F^.�1 ����e��������=�M��Ռ���@��CH������;m���2��ýu�� -C={�_�J����+���=���<��o��:�銽����"�8GL��1ü��F�/6;͖	�q9��=�.Q�� ���=la>���=Y.>Xlm�F�=�?p���'>�|$��@>!}*���;c����<>.�>��=>_.$��H>�<>�Z�%6t<��3�_8�;�bͼ�U>��=е�������>���=�>=��ý���>Ůd>f�>�^�>I��n<t��<�oU>�m6����M����=7�����;�7��/�>��>�g�=ޓ�>��<�]�=|w�>?j�=�i=Ƀ��ݒ�J]ͽ��]gd=Y/�;�U<X�=
���y<��<9=c~�=���=�'B=Q'<�h�#���e=��>���=u�=�\|��,�=��=��<�S�<��Z<���;/RI�W&+<�&������٣='�:�4=+��=50=�و�5��=5c�=���<.�=a��<h��<Q��<�i
>s==xV.>F[=�㗼w�ݼ�ms��C<�#>�X�-�=�>`=3����=���;�^l=�J�=���=��M��yL=��?�0y�<r��<�v=<��S�	=�����=a�<�E�(V4=BX<��ɼXpL=�NK=�̈́=i��=�=��=e�=�s@6�h��<�;�q�;���=u=��=��<�5�=��<�2�<�j�����<1Y����;��Ž�������u�=�L��½|J�=N"R��滈n>Ԓ8�Ħ���m=�ɷ=
zu=���:bF�;��=A�;����]=J�R�����Ul=�kV�*�������誽�����<FԼ�H�1Q���û��ѽ����<Vn�<����s��<A[��ȃ����k��2ʼ��z���g<�/�w�=���=!��=������=]<�;F�̽�=��"�?4����<�䡼�=W��<�j��L^�=H�<'˂�s^����<��=P�!�M�=�;�y�=��<�L�<d�<�ۼ��N��B���<Q�=�8����<'=�S���R���*涽(+��u�<�?���li<�=t��=�	=0N�;��<g��=y#��T	����=��>𽼹I-=kAA>���<�S�=-A�����<����~{�f��=���=�.L��i=hx�<��=��l=`�;P��<�'�=��;=�AI=ӗ�<LV,���׽�9q=29�=����:��<�k�=>�=l^5>��g<l2N���Ƽ<b@=:}�=�����ƻ��)�)>��>�uE�|�v�2��<�m=<RG�9ɪ��[��e}� c^�u��=ä�=������=�zF={Y<��?==���ry�<�ټ����޳=7�7;��=�/��[�������N��e���8=
S=*�I��D�����cBa���Ӽ�<G�*<���=�g�������<3O�=o�8;����~:�<YB��o�=�a��c��;��=0��=� �<�\(=׏μwC<|���<�F��������;����=�o=��ܞ���c=R%���1=����(m�ON=<��=�H��Ӂ��gt���;R,����=���=fG�=a*�.1=v�x�'\�<�R=v<�g�=#�H��9�	��<W�=�N���n=L �<eu6>�>��<���I-�<L4 �p�c<T��= s>\�=z0m=���=��=�/�C�=#�?=&�3<�4�=��Dv�Q9�=�"b=Ĭw;���=�� �4(=�ڈ=nŉ=�{e=89<���=E�>�w�;��>=�����=@s$=@�,��`���G�=��z��=1��<a�%=�<�<t�-<��=�R���%޻��м�F=�M����=1����\=���<��\=!�1=W$�K���t��P؝�y-K=��;��>�K=P�j�{#�=�a�=���<Bx=8'�=��<��<����Q^�M��<�=jI��O�P=�`���L�=�v���=�ߡ=d�v=�
A<�~=�a�=G��9��<=J��=	=$$�=	�<��F���%>�O�=B��#��$`�<������sqJ=���=�/=��8+=ݼi����L�;bi��4�jn=��	}����3=ȫH=_��;=!'=�`C<�oڽ�Չ<�U�< ;���&���v=[� >�����RA=dмl�����=`����Z_�{�a�b,����Q<�w9��jp�M6�<ؕ�r4g=zɼ��T=f�,=k(�=�ټ�/�;X4����=�=�]=ׯ�:U���l����CC=w]<�wt��T���<�J�;.x���%=8��<�J\����=Hμݗ��=d�=ʏ��w����><�{�bޔ�7Z��sG=���z#s�f<�3D�+��u�=O)�]��<����,$=�tD��D>�ʢ�=�G;��F�\.=�}���j<��<m�~�S�<#Od�).<Q��=� ?>���= -����K�7]��#N���������=dY=���<M���x�<4Cļ�	C=��K=�	D�� �<)̦�=+!=?}D�{B�=;8�;�^��U�1���<�<ͽsT�=�h�Y%<5��W����e�t .��{н�W�<I˕��:�<t-#<T��=�s3���=���=�:�=�>���E��*4�v.����<�{�=-P��E�=�9(�(Y_=�l�Tg�������ѺVF�;�<��P����< ��=@I�<!���Ѧ�<�J��4��y#=|F�<�������=�ۛ=�ݳ<�t����k=N�.�G�I��?�=�,�<y�==Z�=ղ=�5���y~=�:�=T9I1-�0轳�<�K��;'�5�9��=e&����[�q6׼J.�=�m��w�=�!{�/�>�}�B�>T >w3�=E�>L�
�m�>��>1���Z>sܤ<��$>�*>�3�<�A>��/�}>��P>q�j>�ʊ=�]J>�<<>*Sa>G�g> �$��<��;=K�l<�p}>��`=��1��=�"�=#~>\���>�L>�5->@�/=�!/>!R�=���<�F>hs[�,�=;G���=�:����&��<�G�=��3�$�.>m#1>��>Q�0>���=�h>(�>̉���Bj;��< �M=k O��.�;�L�=+^Q<�E{=X�y<�!;~��<�%,� �c=�%�>b�<��Q�;�z��V��=��<�Uʽ���Q�4=�+=f�=bF�=�:����>AQ���u�W�(=����b7m��η�M��<�� >��m=4�G=X��=5'=�
}=������>U�=�
A<�q>Me�=Qт=]		��^�=��<�XͽR� =0�ؼ�k�<22b<���<�\�=��>�u�<��>˸,=�0?<���=&>�=��=TY��E�7=uD<���=>����򚩽6۫�]���~�R�L$���+��ZxT=��<���P
�=�_=��^��Ex� �6���P>�6��,)n��!�v���: ����֨�y$����=�Z4��V/=�>.���n�Q�[��K�ݽ.}��x��u1�=��m=���=�I����=L�*[K���ٻz�5��X��#�=��u=�H�=��ս�j
��F�=���=�����Ql��d�=�ۼ8����HE�`��<nA=�Y�<����V��<E�}��f�=�������=a�T=a�0�=`�<�>D=���=�=�u=�֠;�cܽ��n=J��=(V>��=ݝm�	+�=PI�<�R�=�=2���ռ!i�=���<>6��@f>z������t=��9K�<�V�=̲����=�Ƴ�ie>�>I��=���=��H<��=���=|�=,��=4�F���U���7�<=&D$>��<��O��]�=+�;�0#��h�"�!=c�<@D�<��<�e�=d�b=��;����5[�<%q*��"< x��<K��@���o����<^�)��ݓ<_z=_.��՝u=]>9�<��=�~��j&=3Cg=�"�=�U�7�Ӽ
�;/�= �<7&�<�;��<u]ڼW;=��< �»�l�<iE�=&JJ=�Ֆ=:�<���oq��,����֕<��<{��='ٹ<kĺ���<=	���c�;ͅ��Z�b\T���> 	���z��:G�H�=S�b��4��M�=��*�/=g-��c�=axT=s�����=0�<��?��'�=_;�=#<=��O��Ĭ��S� ��"#�=6��=Ύ=�I7<�]p��=�=Tٞ=�{�=�a�<f��<x��=��=���<3����2���u�1X�Ce=����z
�<�=��=�d�=]n��I�[=��=���?�=��=gKI����=�Px=H�f���=O��N	�=0F=Tx�<�8���h>r��<�'�̍Q<���=^K�;��>�&=�����_ؼ͗4���Y��J��K��=��W<�ٛ=j�7��S#>�ߡ<J w�߸�>�0�0�Y<eԂ����=��5=�QA���Q=l�>߂D�g� =aƪ��>�����x��<ܰ^<g�t��L�=dd����=.��=�/�=󯽘e�N->�佗t<���w��>�">�'�=�^�>��A�+௼���<j�>p�=���^���y���=`��P����M�<�>�2>�*�<s�>��?<����Y�>�r�K["�3^�=q���X��<M��'��6Լ{��=���;���=G��'À3=��a�)Ҋ=d^�=��2��1�:�=�P0=�vT<��;� �Z=�W|=��<=��=��= �==�5=�짽���<�s<�Z��V�'=^�7=
\�<&^i�Zc��tT=�ꟽ��=����*1�����Pm�<���<�Y�=�C�<���=t�ȼ�F�<䮐:�
�<&
�X#
�:|=��<�Y����,�����BL%>��$�?Ӎ=���<@z�<i51=��=�=����<��p<&_Z=�%F=I<�=��，����r�>	����E�<>�g�HC�<Cs
>��!>��=�/6�l�}==>jG��}Vܼ��*=�4�=^��=��=�e���'�����=&�F�~�=��#��਽ȥw�!4w=�zH=�Lk=�=|�a<r��<��C;
���ZP=d��;;d�=���=�`?�
C#�a��&=���k�=�B<�"<��=A]�;��7=�{=l<H���@ <8�l=M��e2x=�=&<�5н���<L궻fqý��=^�=��=\>�=^�<�t�=_}��h�<����><�t=B�=�1�=���=ak=��0=��{=p�<�6�<v�C=!L½Zs㼨_x<]�>F�t��n=�}����=)�=�6�:����%:�=|.�=$L�<d}�=�b=J��<���=Ѕ�=��;rԫ<��۽k�<- ����=���<Y�> Q�=$�=X��=�����[��T�;@��=�s9�����4<�p�;�A�<�I����x������c�;w�����w=x輳���z����=?4���!�<"���@�<=����y=(~�Ē���x�<�=ށ�5w'=��=����-�F�қ�<�!�=hVa��s���܈��¼X��;U�<�c�������nD=�B���w�S����-X�|��Ҵֻ4�<�|P�2g�<+�B=i ,=H�=o�+�
Ӳ:���<�X���E==�R=)�=��;�d�=���������=�3=o��=�ۺ�Yh��P���=�5w=�} <�$>�U�=�7�<�1<`�8;c�:=��=��E�z,p:�=͔<
�y���\=��<����Z=A��;��=�ڜ=C�<$Dý��<=zl8��礼p�R=��=����4�=!��=}>�����F��t\=}ٯ=Ӝ��:�=M]e=������ ���=���@*��楻f�=P�=���l��;cA�=��[=������=���=o��<���=F�L�K>�n��#�=��ؽ	
�=�����.;�\�=���=�Z�<� C�89=Ve=��up=��G���9`=D��=y�2�*b2=����o6=0�=j�=��=�,��S�u�>eÄ��B
�@g=��޽|XD���<��V=l�C��t��c��=���=���=�f�=1��=���=�{;O�;�"�g=%7<}6K=�z�o���V��<��=��=:둽��W_H=(�=#�C��r�={�=Z��qY=����B�ncD�5a��r���S�����`�B=�?P��
��t�=°?�<Н=y=�=�=,)<?�=̢=뤓<���=���=��E�C`�=ߥ>=�y0�tҞ<��V=�q�=O-�;�;�<�.�:y��(�<�#]=��<[
�=kG׼��_=��{=�i= ����>GP�=g��<�Z�=�X�<�O.�)첺��=���=+�d=�=��<�ؽ��<�7�_�O=ۘ	����=�=���Y]�=��<e_<�O�����:E=>���=J̑�X�ͽ�s����^;et�=�=�l=ф�<Q�<>/�EW9=���dt<z��;�w�<`O9=�zF�PH�=<M�=J��<�F�=�H7>��/��C'=�X=i���n ��;=N����=��=O�a=��<�=�'�B�=!����z�=�Rv=]C��Ϻ<�ݻ��5=�C>�ʵ=�`��#��=X_=>�=I^�=�&�;�a=��<��s=��p�׃
;���;)�;�]Z�h;�*t��}�P�<�΅���=����)\>1�=�f1=�?!>��r�+g9=��>�e��IX�=�I\�h;��U�U8ҽ|뛼�[>=z�<�:>�6�=�er�ך�=�p0<���軺=�t�.E|;}-���PF>r"�<0:�<E n=��K>��{=��C=̱��i1�>uYO>�d>e��>#n�B�Ƽ �<"�4>)WϽYνWL��~��C�k�'$H�ｎ�>s�=�!�=���>�$�]�j�J>��<H��
�"=|CF=zռ��=	#�_���_W�:��f��%�<�آ�/u���v=�!�=?���m��^�:��^ <���<Ӳ޼�~	�R��;�
4=�+-=�P=.�<�E=C�|����<b�=��������̤=��==٠2=����{<�����-�<r��<i��P��<~y<�����)���Ǻ��ҽ�\�����L��bi�����<�=PfO<��J=S#Z=a���K=�u=����_�4�'�N���=c9�=ⱑ=���<�՟=?�=|���+��;y�<�#>�U=��1=�>�=u�(<�	�=#|?����@�.>�bнLO�;�Ỡ`�=�󙼉3>��=�<OQ�=&3'��C>��[<=���Bi=3�������G��1A>�e�=	J<�\�<�a>��=^m >�Ҁ�R�>�>.��>H��>�n�=W�<`�=<�\>�h�<FfL��{��W~�<]�<����i���)�>CF�>�o�;��F>:��<��u��η>��;��ļ���Ģ_=#�<0�=��r�v�H�Ah���P�<Ż�v=�x�tv�����=�ݧ=���=#
;�t�R< �M�N�=��=��=�ާ=�+�9*��~�=1��<��6��="�:*x����8=U3N�2-�����=$mr�F���^�>6M���t9=6#;�n�<���=����ɼ��B�� R�� �+=g��=x
����.=9�;��y:���?��<���<`��<o-���=faQ��XK<�7L<����=�Z=���=�#���.�=H�L=G�=:��c=��<=l��=��o����=,��=P��
t4�N�=+�+�x��E�򮕼7�=�
���&=�#��G�h���<vy��Am�۩%=��=,	�!@̽Z��=�H��ݗ��Ӡ�=�$���O=�����=0>�L7��C?��b��fL&��n=��w=0_H�����392���=�@��FGJ�P��=3\�;>�=�������ۂ�_�s����=٤
=Yxýۺ�=P>y�^2�=�s=R�=]}��˸D< ������=DX����<�iC�CQ��b�<�Ѐ=1����=���<��=�V�=1��;t'?>&��X;=K ;��=����ԑ=���=�6���>��Z��3�=h��=Vu��9��=�1�<�ÿ=���=,8�H�H=R.�:g�̼��~=п�;�����e<D��<9�5>K[h=Ӱ�=b����$��ʐ�JU�4�<m��<��u�5C=�L~=��=���<>4���N<V=�/=��=�}���78=9W"<z:> .=.�<�b= U�;�����.�d��<y�<
gr=�̟=҃�,�!=1�v=#��<�!=-j����=L�=^�c=ݿ����)R>{-T<d�����E���>���C�<���� =Eػ|Z<L
�=C�=�$�<㊒=K��=���<�V>�'�=�=�=>�=�6=��=i�=�l=L�����]�=˞��I�<Nz�:�v=���;5�;����ʲ> �=�-��q+�b>W=Ù���+:��=�W%=װ3��*���\��-po=஼a���ˮܽdt��q�;��E�ܩ�Yu���2�<�	Ž��9��ӽ��߽]Q�Bz[�sf��~��	���6�н�Ӓ=IL�=��<J�!=k=��󔽵->�څ��O�#���q�>�8�Y��s�ܽdI���=���;{��{�<a�ֽ=QJ=���<�/�%�B���`;�A���뼭��z:@��;wj�=�$ڽ��ν���<_wX���(���=.���=MM���il��T�<z	����<����=�	�>;GX�
�>�`�qXR���
=TE�ī޼|����=�Լ�%=��E->k���<��=uw�;Dr9��#=4��;�{�<5c�Mi��-)=R�S'=0�нM4:>�̞�F�=��<>MP:�_�̽D*��2Z<�:<��;�U�<�b>=i<�=����	=R%>��>��m=�F&>�N5=��<7 �=M�3�V�=Z{�=�E�i�<P��Գ�=�Լ}%�=q��=�����D�<�����U�=��=N:9=�=�{>=�i�=���=^"�=��">Ow��%JY��^g=]j�<���=\��=��=��1=	g�����}Ǟ�z	>'��=r�s=!'=%eX=�<<y!�=ϥ�=G��=�=:	>���=�gJ= q�=��#<�� =��	=O\
>�fD=d���C�
3�<</k>�m�I��=p�4<��E���=&$�=�+�=��L=��F�/?���O�=A^b<�3�Mh-�[�	>IW����|<�>d�L=��1=DvＨ|�=9ˣ=�B=�=��=F�*=�n3�O��=z�=��	���<qT�<I�5=��=!�ѻ��5=�G�=8� >B��<u��LR(=��޽��=K�=�=an����<�>0�t�S �<���=$m�;���BR�Ռl<��H=��u �<ڼ3��N����c�]s)��ѕ=��H=��=��=�=j�����
���<�=��=d�<�3�=�->��a=�
�����=oMe=b!��=>>&�t�����w��=⫒�Ug���5k�)�M<�-�	��=N49<P��Խ��t<��<��{;Y{9=4�9=���=�>��=\��< ��=퇔=�����uY=�L�<C�=��E<T��ҷ�=�>�=�D�<fEZ<�L}=w����D=%Z*=�	����= 8ļ�K�F�+=oq��X=���<�a!=�s�<ӫe��R<�y�;�K���f=�۾��h�=I����7�]�<L�(p�:�n��C[�L4Ƚ���yW����1G�;�a�������:ű��!Q���3�;ꣽ��g��k�=�/���E��!��W���\�:�������r[��4�A��=6�=���=(�?��1,���<Q-����=��w���Y��ˡ��f	� �	����<��	�mL(��6e�2�����K�"�>��_=�xo��s��Խ��=P�>�0�{�ŽW���zTս͸ͽ��w۽(�ؽc�Y=�˼Or�=+O�<#���9D˽�xu=so�=� �;�='�c�m�=?ܝ<	灻���=d8�<�G0�#k����=Ο<�����;.=a�=F�	�'��=��%�f�?<7��<�<�=<�
���,��Ǐ;�X����%<���=��S=1bL=�@��$�
=!N����<GcW=�4=����k�=���=�H= �<��=��A���<��d���=ݼ�<o�T�C�R<zn�=�sC=��=���=�;��a`=�w
>��W�H�=�.]����=Xz�=�Q>�3=�)2��C�;�>���=vȼ7&�=PY��}��
N�=JU�+��<�~=l(=9��<sЇ���=�+=⠄<$I=J3�=B�=��=Dy=3�d��o�C2�<@����G=���=�W��0uF�5m=�	>Q]�=�Q��F��<�;p.�s}=��=�ǉ=)����4<�$��@�<�`���<(�=���"�G�A�����O���b�,��<��y=�F޽gn,=J�K�}� <ȓ=�F�;!k~�4�<�^�-<掄����=>������.��JrM=��$�6�<¬�9�<'¹=�a�<�9�;L��"f=��k<�]`�9�]=�բ�ؤ�<�]=ZV�;2v ��/{=?���%�%���>)ޓ=>Q*=J@=Z��W!�<(�ļP"W=��<L^R<�@	=r%�����)f���l=)w�<.��<�b;=Mg9=H�=�Fͺs{���J<'��=y5�=� Y=��R=א��.	>�Z8�o�2���=H�=Pj=��5=EW<�#?��4�<��K��;wM�=`K!=M��=�0;=���kƔ�-��存<���=�����WJG�SL=�!=2�;�)l=�x㻑��=k��ǵ�=>$<�;Ż���w����m���Z;=��<|<֌/��=�$�s��6�W=�H=^����=$���
�;a��<d�=�m�TR�����߽�=,�[=աf=#�\M�<P��=��<쐻��`�=kE=n�<׃�<:&��{��<��<�yU�fS��2�*��cn<�dv�Dr>�v;̹	��/�
HY=�˼8����������Q�<b���O�<�Q��-X)���X���(<R��=7c��gxI�}�}�3rx;^��=��=����(K������&�<��=[[����=P!]=�=/�>ؽ�<&QK��j=�U�=�n!:
���d<�K�=#�<o�H=��<忉<��a=�=���� ����=��ʼ @L��͎<�L
<�.��zͽq��Fc����	=җ�==������=C1C�Za�:��=�B�!��x�=�e���!<1��<��&=���=�T�� w�=�-��Ώr=�#���p��� )�O��<��=��=3Q>��;�n����@=en�<���<a�=�mD�1"R�Y�=P��=���<I�b=N�M=��<���=K��=��v����=!I��PλeO�=�F��W��,��=�P�=C����=�ݠ;��Z=E�<��M��sZ<�∼C��=^O���К��P=M��=à潀T�������/�/=[���0�콂��<X ��k/,=��������Á�U��8��93݂;�3�;�n�=�<>�
=U��<�� =�����м˔�<�5=��Ž�׽�~i=�DZ=9p�ۥ����<�G���y<o
�3��;{�+�;CA=l��/������@�;"��=�j=hn�<�T��-�<�5��ʽ� �<�ʲ<E��s�������w9���Hcb=p����":�^��;n=y��=��<1	����<p�
>ɲ����=[Ѫ��QR=`��H��=�L.>z4<di�<�
�=H1�<ˌ�<�'>lE�<eJd=��1=K��<��s=-=$�3>�қ<��<'�i��==���=}t�=U��=�˛�"�F��#x=7%>w��o�����V��
��k�=4�>��=��=ϟ<d�=��<��<���� C=�P��ؼ�|�<�eN���=�H�0ā=b�e��I����0=� 0=�A�v�Ͻ��;��<t뫼j���c�v��4�=�D����=�^V=�X��>0=�����}瞼~�8�P��-����M=Q���=@N�<$?i��m=d�2���X��S<=�R���1��k�:��=�2=�F��噆�Ʊ=�e�LT@���ý��7>��=.E�=b�w>Gc������܋=���=�V�I����w�;�Vu�����a*ҽ�`J>EH�<h�t��g�>�dP;˪���=��/�1 ýNi=^�}�����:����,�v�Z��{=�>�8.�=�> mk�<�D=%��='�<�c�=�D����>�њ=�S=1�q=�� >��>�\j>�y�={�=	�>�Rh=��G����=�����ꑼF�*�!�>3�>���<����g�u=~	��!>�\a�)W�=u�|=�3>��?>C�=MC�;�I#>�4�=�4>�����/�:��@�^���G�=o$��
w=|�*>:�=�j>��V>>-�=�v�=��2<'(ڼ-���U~����<�����4�5���ڽ���~�k�ɿ'�\�=8����Q��U<�j��An=����7=���0�X �l��������]��r����^��)?�T��՗U�(>Ce ��f=|���w�w.��=׹���(��G��vK>�N9��pj��;��a��EŽ�f�0븽��Q����0/>�ϼ�nK����j�T�D�G>��W�k��G/�'� �]��%X�`���HZ����=�#��<Lf����<��=+�.=	�
=�R�=+�H�6/��fJ���;�a��=�=6ʉ<�����:�=��{=�Ļ�8�;�ڣ=�B�Ȅ�==E�~=DhZ�'��=A}d;3�2<�}� Ë<R'~<�Bw<f�<{_�=&�7<���=���;�"�=x�]=��;�{<J�=���=�n��Oa���	;:d=@u(=��l=�� ���<��9��<n���S�=6�<��="'�<�t<�:�=!�=C�t�B����P��$�滮�<k ":��F�I74�g���Lp�腽�N#5��n��!M�JK=���k�l=�BϽ�:���a���Ͻ��	�ƛ�+�����>x��`�"W���e��'!���>��=z��=�����W-��ힽ���=����@����2��>�2I�UL�0c���z����+�<�fa�t�����">t�I��@�=��t={�A��M�=�x�����K:;����d�2Z��0�u�9t|u��ݻ='S,��!=J{�=�aѽ�il=��H��4<Ss�<2h�ڌ�;��<[��ݜ_<7f���(;��:���M�z�������ﻢ�X=�*�� z<:���O���_��<px���{�<Zφ<]��
m=��:=�Q*��Q�U��;� ��!%<G ���	+��u�<_8i=�@`��#�������޼�.\=�.&=\�<��h��U�Iļ]���'�0=X��<M耽]nS� 2N�CM�s�4���;T�=�D�9��<=@�Y<���垺�-}=HN���!�=�C�=�d���� �;S�R�;	��=٤:=1rZ�����5�<�:<�{�=q��Ϫ=�]U=���:5$ȼʷ8=��5=���=ZHּL>�'���Ǧ<}\�=P3:>��=xzۼ��<h/�=_�v���=G���f�<)��<��]�oS;�x=���T��<�^#���=����%����i}�cr	=��M������Ͻ��/��>]�=���<���=R��.du<mb:���=�j޺���뼨v��	���Y	<�����'ѽ!�#<�=2�8=13�=3�
��.}�D9�=|��;H����>�b=v�=mռ;�m�h3<�2��\Q��V�=�޾�iʸ<
ލ�ǚ>�����p��=�z�<�V,>���� �/�ok
>���O��=ZZ
>�OR=��M�Bv=�$�=��P>������=8�����߽�<=�T��W/�b=�=��=B3>el�=q��=�X~;nk�*�4�į��/$=�T�i�<�մ=��;�7eP=�rR=�`�`ӽ��o�����?�4=V
���겼�KZ�j�=И2=���=2�=2�=�ql�4�b��>W�9�ˁ���s>���<]�K=Ɔ5��_��"C���m>�{-=Ϣ0�C䅽���t��<`Mʽ#ͭ�/��=�f*��P�=0��=#>��Y�⼅��<�՟=!W�=;�j���'�kU<2���rӦ=$�S��?�=��=�f=�=��=�.<��o<s��<��S;D��$�{�R=r��<��u�M<��7�#o�<paȽQ4S���=��F�rӠ>̪c=�#>`�=�L=0K��{=+���k�;�W=�m=w�>1��xd�= +<=�=�;�ǐ=B�1=�ƽ�<3�=����D��<h�>P:�=���;����t"r>��!=�:	=�x>��ʼ;n��� =�iL���N=��	=W�<����׽����Z�9>�b�=��=c^A>Aw�=�Ku=>e/>X'�����<�v�>��:�B�>�N���X=�Fa��ݘ>M�$>�>6P�>)�>=�O>GC:?N�<�_�>��B�>S+>��<JHH>a0!>�
�>��>���>e4��?��>11>Z�>�X6>bg>@Düv3+=V��>!1�>g>qk���>8��>��>-���}��>�t�>Wn>���>��=�G>��=�N�=��i=�:8<��Ag=ˀ>7�<-��=��?�x�>e�_>cR?��,>g�/>\!�>"�A>eR��"�|=3���'𥽅{=Z�;�v���S��N����u=5"�<���<�������J�����<	{=r����-�4�<v(�<`�G� 2�<�f"��d"�\J�vv*=v�"��<¯�=�V|��eD=�&ƻ:��;(ܼ�[N����EM�=�X=�h,�3?*=�9z�yڀ�>#��=Zm=��T=�)=L�<+j�=�%`=���=c�=�Nw<��->At��?#�;P�=�����<?�=�D��b�*�@��=�ᘼP)
=7�=���ع=V��<�n��ͣ=Z뿼9?|��=w�z���e��=��G=<dX�4\� ��=ٞ�<�1"=��;�L>.>Vx=X��=�82�>���\�|�>�=є��&d=UvU����=��=4���j�0=!�==!`=<�џ= p����=�+�=� >u�><�=JzN�o�=f�=�]=҆q�׽�ar�O)�=�_Q=�r ��¼�"�=��>l�=>.M���>���=]��=FD�VRv=;��R��<����3�=j����=��=�ĕ�s'�<��\=48=D�>]��<�O��������<��=c��=��(���~<�V�<τA=Xz�<� �<{e<# �=S%J=K��=�	=���=2�����=Y(w=.��<ϖ��=T�<J��=,Z���i�=�����=�]�=C���K��=���e�=���=XoM�*���5�p��	����+=�ذ��ү�q�8;,��=�
>4ľ<H+>�!>I��<ɢ�=��Լ`s��Ԯ���=@�=�<�L=~��<V=}b�=�x|<��=��=Y0t<f��=g�P�|n�=�u>>�ܼ2�=� �=z{A����������=��=#�=�Z����=�����!�?�ļ�ý=���=��������5���=@'�;��I�ԩ$=�j<�� >E$>'���É<<�F���g�ZHG=W?�|*����<�h��"���C�=_�>�<��=�I��0�<e����'
�2�
��;i�S[�=�$t<=˙�D[3<ӼS�ڛ��b=`�<Ϳ ��+��Z;��4(>��#=t�H�Z��<sԽ�
�=[��=��=|���_�=��H�wY�<\1�=c">&+Ļ�K�=�d�=��;����b�\ﳽ�&T>@��=Cw˺g����ɼo��:6u|<��ݼ��>=���=_E��F >d(���=��+<���=�{�=z����}�o�Z��l�_=��������<��/=7V�=��Q=>�=���;�%_��,�}���L=��k=�G��#R��~=-���DK���d��'��9���
	-�~{t�vؽ4�Y=�A��|���b�7
��ץ<|l�A�\��Y콍��=3���6��ê����*<�Yy=p�	�L�#>"��߽ܵ�l{��(%�>"��`�l]��a�=����l����$*����\L����x���V��̽��8��RI=m&��]��=�ҫ=�蠼(�=/ؽ�l��f$�����&=���;!b���ݝ���:=A뽂==�p���P�N>�"�_=[�=��v=q$�Ѹ���:���'�`>����y[=o{���z��bm��"�RG���>��<�݀=�R=L��e:L>	pн[Gy�#$4�)݀�Z!����ֺ�>���=^
��f];���=��=��=ȐѾ��1>N�>��=1��>��L����E>ո7>c�g=(\�:.�=iL�=!9u��]�G�.�y�0>'�����=�H�>���=��E���=Q==}h��½q���n�m(�^�==�4�E>S�ڲ</\2����=ر��f�Z�����.�;Zj�<�U;��μ_�<'>@'�<��Q>�2m<������<-�(=Y���X=}�=M]�=,���]�<>��@�3>'�Y<F?��3]���d�;ȹ�=5��=,�:�{�P"��)ֲ<w=r�c�<x�'�Ӗ�=�)�=A	�<�gu���H��4���vn�=�C�x��hݽJ�=�� =�O7=�M�=m�.=R�}=Jw��.��S�����Hӽ�aϼ6�7����=`������:z=�����-�5�=��*���S;�	�=����{��=t�=w�=�X��߯5=F�=>��=tkc=���<��$���z=�V.=*��V�=+�>�-�l:��:#4<�R�=�g<�Y�<��8�������}���=�9@=�3�=��2��=�(}<�ښ=?I�<���<�QT�N�T=����{;!>��<7��:8��* �=|�J=7�=Fa1>�1��
1���˽%g[������<�p/�3�<9m����=go��H=��:�Ե�#�J�_�=�Ī=�Ƽc�<�pջ Q�᜛=�Y��˽hp��KI�����[����p�u�H��/V=��8�ýӯҽO��u����<��"���ۻ��5=R4=��x���<=�;:���<[� =P7i<��ֽ�i�<u#r�ڕ�<���=C;Y;��!=���Q�<�<=sl���ݼ@�i�w�=?� =�?�<#'z=�b;��L=/�=�D�<d�&=��&<��D�<�1>�l; E=���<�w4=���=�(=�[
�B'�=r�v=�7=8�=i̪=˱|=�<u=zm"�ڱ_���^=����m�>&�;��=-:C<�f�<�P<b/	=�AƼ˛X�� �[��<�r�<�zw=Y��=,�Wm>8�e=��X<gD�=@Y�<�0=��o=��8=��>lD�=���=��-�.Qt<�'�='.��yP�=��<Bj�=|�O=���=d��=�
>�
=�$��#`6=�O�;;��;��
;e�<#�ڼ��<2`=\�=7\<����+���;u��-m=fޝ�yy�<@l����<��ݺ���<j�\��]<V��o�=G��<�=[�j=��+=�*n�)3�<��=�%����k�l��<��=�H?=*������=�R�<:(��0��)�=elX��C��f�F;�V;V�<���7�ü-�$<�,�=��<�w�<r��[�)�د����d�<�JI=k]i��Ӊ;���o�t��`�s<=���}��<^��NF>tyB=�<����=m�2=�X'�D|G>Oŗ�K�<戼�cO=NЫ<Í=�"4=���=Ʀ<Ц����;��ӳ=��M=�7o�J(�=�OF=��LJѻ�<Af�=����찾=��}=01�=��+=��?=H�ԽL�=�=�=��=պy=1R��ڧ[=��̼��t=��|�>Q�յ�=G��:/a�=�5�<���=��==�K����X>W��<}C�=| �C��<�C��Đ�
q�|B�<�1�=i��<�>>�R����������uO=x�i�����;>0��/-C=d2'=D��<dp�=)0�;6>?�������<����g�=Y�i=�1>�d�<��8���,A�=h��;�5?=��=��=�~	��v=�A�=>�
=���D!=�j>��!=,��=��\=��1�ˁ��>&p=t�"���tM�=�"1=|[=Ń�<�1���J=3��<��p����=�>��<>.>�P�=R.�=2kܼ3oż����"x̼2���Pj`=.<ػ��"=��1=r���S=%܁=[�>��=�t8�hrz=�Y�=K�Ĺ�(=7^$=5F=�)��T����<�μpHY<��i<�!�����ʽ�����-��j��DG=ȗ=�Qj�.�:M�<"�-:˪�<��;t�;hh<���<������	Z�=Qw��	5���;�-;=$����>����p&<Kp�=!�s��ܛ<ԋ@��:��.�����b�
�=������{�<�������<}��ѕ�Wm�=�=��}���f=�Yu��Yz���p��l�=�7��J���kn�<=aD<�#=��?�R��o���%8,=5�m;����7M<��˼��V=J#*=B�l='���䧽�-�<��3<3�����n����]�ĉ<H����<3�c����=��<0m =5�H�����#�d�)����==��<�J=�ʹ���'�A���<�<��i=cL�=P2�7ʋ��O�<�ȡ=jf�=ֈT=�A{�\�軻�ļ�J5=�!=�(�������=c
.�uP8�X�<��9>�Fc;�!g<�Z�=)��h�:=��=���p"��/�<,6��1�X= 1 ���A<�؈=��uI�<�=";%���1z�Ɨݽ��i<�G�e��9P7�9t�Z���;�:��Ch��
�?�~�=t�ֺ��=��,=�<�ټ�@�<@+�==�켅Ĩ<AcT=�JL=j�7=e�<���#=��d=��;��96E�=���=�*�<ĩ�;��o��w����I:�9��H�����e�=Q��=�sg=҄<��;��}=&&?gF6�f��=�jQ=��"=�:k=B�n�刡: ��=5�	>��=���=tY���S5=kdx=�긼��*=EּxW�=�R��Q��>��;��E�� z��>|�=�J�=;�o���>F�I>|�X>f�>�{=�@˽n�];4)">�SŽ����&���.9=ϔͼX�< �I��>�z�>	�<��>�h1��Q�S̔>�w���䔽���� �;�h�9}�<8(���?=Ί0=��(�0��=(��=`S5=����	mi���q�kڣ�V��;��>�M�<]ǲ�|�R>Z�=v�=
�<9����t�=�q��Z��®=j��*μm$�1#t>?��=�߳��^���S�tD�=�=cfx����=�>}iz<�
">����%轝"�=��=۹=p3h���*<�o�<<6�Qk�:4_�R& �E�<����t1>��=Z[�=0_�<yp�=��ɇ<<�+=yFp=�����½-�ռPQ���l���
佅�=������ �)�t�=�>=%l��]�(>*d���﷽��z�/���&�=����\;/��t6Ͻ��,=������a�����=�gc����<�ov�;��ɏ���2�Т��"h�������a>5�ĽN�<	6��^6�E��J�;hZ����9��#�=���*�]<���=K��ܝU=�P�%��a#8����ԛϻw/_�AH:��J���׽<=�A�<Ra=Sq��#�<���l�<!�{;�%���X<��6��	ֻ��==%�A�<R���꼜�I>Kl�=n=˻ܼ�<�l=�9�;q>^=�+=�=5�=t�@=�b�<��q�׼�g�bQٻ�Rk=��6��-M<��J=���=,�[=�U�;�=T#�=PH�=*3�;����z7��Y�v=�ߔ=@�=L뷽W�}�������������B��<ي=9v�=rٌ=Jl�<Y�=H�=4�;"��mt)=�N�<�{n�_P<=Oq�=*һ�`=�B.>j�R=�~j=����<�q[=+�����<o�"=��<4�=��=�=�@��&<m�φ �q�(=0���_=;�=k�.�7x8��Hڼg�J����	n���P~<�+�;�R���M�=��u=˼��Ah%<�<�r=ȯ�=7S�=��L�F�>�	?=m֦=����=��=���<s=�b�;�����:�2T=��m��)J=�:;=��.�*�_=w�<�W��2�w�i,�<��ؼM,o=��=�p>�)k<V��=B*�Ҧ����ɼ��'>�ε��#�a�<਩<,�<�����>@���
=#��=�[k<>��=��=�?�=F)>|�=F5ܽ~��<x��*�ؿ<V�=���r�=�8�<5�T=CA�<GÝ=�������=v]=��=�>=#x�<L��{��˽�XQ=ASG�CC�;���=��m�q==%�=�=C���h�c=pe?=�����u�0������J`��A�����C<,�.�I8�=E�{����z��3�G��N<:';;��=X��<e䤼��<W������U]ý�IY���>�=N
�؋�L!��]�=�?��hN�L�����=��콚J�<��>�/�;�����=�X�<]v�p̖�ړ+=!�M=46�;*.<D�m>�UK��I�m8��&=�|ʽ�i=����(��ֽ�ӽs����P)=RKW��:��j>���<&�ǽ*�<�.���|�=�䉽�ց���	�.3,=f�w������帷/�<q��%)���E�O9=~��=�ؚ�cğ=J�A�0&�=�2�=J�=X��=O��<�氼cdx�������'=ܱ��l�=�ػ�'��ן�C����C��;��O��]���=%��Z�=��r<��K��@�<�Ћ;
�`=t�=���=�l�;�
>���=�ɶ=�2׽��y=�j���R=��J=�O�.�_�I��=���=p�">%�=q��<!�=�r�;S_J��ˉ��<ؙ����<,5=��;30>N��;uҽ�C�:��P=�ޣ<��=%O��^<����;��=T�<��=����#� >+O��v���k�=�jt=���<���<}�k=q+=:�Ľ�a;ܔ���͉>'Ն<:���*/���=�$�<�����w]�R���^Uy<��<�s�=�v"=�␼��!�4$=��=��ҽ��#=��<��R��@���4�1�g�%=���<�TY=��[>ڍ�=�ʺL�(=����pi���=q�R������w���W��[��^��<��f=뿐�/(ν4h��	\�)�o�Tv$=A3�=��i�}��=���=��<�����o>c-=">�;)��=��>=#"�=�b�<�^>��-������g�<�����>�I�<�U�=�-I<��½��<��;�)�^&{����<IY�:H+<&>�CĽ٣=��̼ 1�=��]��	�@;W)�$��.�wd���c<�gu=��=�SY=>��=M���U!�����;/��=�DM��I;c�N�7�<�G�;W�E��^ =,��}��<g��Mj[� ̱=p�<_�h;�o��o=�\��{P<�������<���:-�=v�6�P�T<�ʋ�/7,<��i��C���g=�����j��v":�1�<;�ڽk͏�$|���6���-�=�ӷ=u/=��={x_�xfj=�4��ub<��"� �N�;�i�YN=���=�ܼ�e���P�h=���o��f
��)<=�����<U�[������
9��߻��!=6P߽6�^�M�I=���=_��=�>-��;��o=R�g���,�Y;b<����y�X<��^��@�=ء=���;H�=��>m}�=K��=�>�f<p�9��D<��=]K�$֯��l�:��	�-ĥ=�������=�a��_<�y�=�|*=���eׅ=��=��-<:�
=����	=�U���g=!�>"L��g���Ȥ�� ���=g��z��8^>ed4���=�9==O�>�}&�`��=��=��h=��?�CQ���
�<�/�BG�<���<��;j���˼��h�(`ͻ=4w>@=̳Ļ��ݽ��m=�y��O�A=��i�&�v=�c�<��v<�Z=�E��q�=�����E==��Ƽ���A��<z7=��X�T��;�>�A'=P��i=��{	>g�>qu�=-C>2�W<U��<	ӽ����8�<�X��o �<	�5��g(��۽�Uc����=�C=�hY�8>bI9��ӎ;b�=�h^�tD ����S����սt���0�=�3���8P=2�]=p`J�;�h��弎�غεV=1=��6��X=��<4�<{`�=��]=R*E>�W��r	>$Ȼ��=/��=רA=�n�<�=�>�"=¼������=	q���q<�y�˻�g�=9a<��e<��/껪�=��p=��x<P�X=�/��'��=�Q�=��<: �=L��{��a��ϵ1�����������=��{��=�w=�?�w'D�,�;�E��)� �Bx�����=%`�<�C�=����Խ�<\�<B*X�u�B=�b=��=^�D>�M@�K��=�P-��f��n�=�d:=��?������{N�eߡ=�y�=��f<�D=0�=�M=���f� �@����J��,;<r����8*=;�c����=���=�.=�[�z3#=~K8=��׼���=A8��j��<�'�=�,?=��=�+�Ŕ;;�i�<�5;��m�d6{�B��=��1=�T�uw>��J��`=��8���$���=]*��5޽V�!�O��<�c⽯����&6�r�4=?�|����;2{�=ڀ���t >���� <�.��O;����6��{���;�^�>���<�R;���=Q�4�UQi=n�E��'��̺��������=�Ͻ��>�c= 4���=�̉�om�<C��x�b���=��a=�ҵ<T�`>���D��gU�=��*=���4b�`�νa[��T���<��>V�9^g<h~�<�q�=��>���<�Ž^�E=ނݽ�����G<��8=��<��'��'�<'�{=�Lg���<J\�=�-=kYû�@�!��;���c����+���x�����=�~���EQ��e�W��=�����I=|<=���=�;��;��������Oe�;��2=�+��
 Z;�p�=�X�=0���w�;=�f�F���x�q<��<}	Y<k�(�~3K=X�<^`��
�W�-�B�V=��$<H�º��=��= U׽�M�=�k��[�=;�P=�;=�(�Z-�; �U���)��s#�u��Q���#�<5�����B=�M�Y>�����!*�:ɻнe~�>;&M�k+u�~Zw����<�]�t�=eu�����>���=�.����=ڮ��l<�;���ܽ�qq=�����V%2��z�>V��=�By�Rۊ���7=��=�X߽�i���iJ>��=Q@�=��>��޽t�⽔@t=��[=+��	���s����'=/���z�\���g�>\�>�Ҧ=��>�,�=@U ���=�N��Ă=�r��z��=[2y<V!=�W�<��`�{)�#Ѥ<+F��?,2�Ǚ-���+�����A�<��<�.�A䧽ԕC�F��-��=���7*=�T�;��o=D <�^K�!�̼���<�mk=�̔9j��<#��=��<eFp�����P��i�<J��=�f���o�<�1��o�޻
��)�=�R�=�F˽$���<=%��37=O}r���e��r�=V)�<���;�T�/ڇ;;e=~e��j��(�=�Ko��,�=m����`�<�������=��.LF=�2����<�b[<��>�ލ=�Ȉ=���;\Q�=�H�=�[W�ג�=���<�*��Qq�G�=�ns�#X&>�%����E=���=�k�<���<�<�=ɴ��Lc<A��<`т<�_�(�
>~s�=� =��F��.�=�<�7�<��<�e�܍=(�i����=���=���=K���k�Fp�=M2��Ϙ���>r����ʟ=�<��U�,��='�c�,{9=�>��>�V<L){�fEB=ԏ�7�.���<Y�=p�м�~;=���'Vw�H�=�]o���!�z��=:��!�=n-ɽ�)>�P�=�9B=�<o�>���=[�=Tc�=�;N>��z=BW<�'>�����}�:=ή/��|�=14���{��[�=�{�=(��=C��=���!Xm=���<(�=ڊ�=XV6=:��=�k����=��<����^=�f�������ur����H�y^\<ń�=O��=����p<=7��J�K��̼��H=Fܟ��s�/�<ЄD�4e�=3lG:��<��<음�$N6��|�=J�=��^�68�=Ŭ�������^��=S=�am<�>�՟=��=F��<E�=ќ�=wR>��j=}�;"`&�y6k��p1���=es���,>z���(�=.*i=��=3�N�p��<�3S=��p=�R�:4<�Q�=֧�=�,�=Q��=����z㽘fټ��<�2*=:d�Q���kv=�z];��̻����U��=�Q�=��_=�d,=��=ͼ�<�zX>1z��
>�
�;�Z�=�=e�@�->5"�=�}�{伨����i��39���=v��=���!oܻI��<81�� �������Q�5�,M�=C�<��=�>�=�ԡ�OgC>2��=$�=�(��Fǀ<�l>0�ؔ	�1�I<�m_=�[�>L��k׽���<Q^�v��=�(&>�Ò�p�	�X	��=�=����:w���Ck>"��=qѻ=|�a���@#F��	������ĝ�r�޽��=���K��&ʒ�%D6m���	���;���=�w�=
 �=�ޖ=4�=���ѿ>���;1��=.����f����=N�]<)�	�[�>>Uqa>U�=6Y�=z�y��RB���*���<�9�����E=��Ͻ���>���=��y]=��=�a�<%b��d]���>ۚ><B�=d@�>�}��+�\��<�D�>�=8n���ǃ=;K�;����@��<}� ��QG>!>�<Ef�>F�C=5�=$9�=��?�К1�6��=�i���9d�+B=<�Y=��A=by�=��=H/����$=~=�d]=:��=���ۛ�]���x�=}U#>���=U�<K�=E���W�<���=�S�=Ѱ=�t=�T�=�>-=�m��w���\	��w#>B�߼4[��>�.�L>*u=wd�<� �rcn=�>����=V�!>x"�<@�>�_�<\m�= �+<����G �=��G<τ�]�:�	���}�=�>��d=�Z=|�!:A�'=xX�=�Ц=<�<�ڕ��l�åv=<p<[�=)���{�=��!=��y�b�~��JR�wPU���5>yt��e��;^��d=KO;2s�=�Y�=�`~= ~�=sμ����ЎL� ֠=�(<R5>ST�<!�X�<!,=�ɽ�ʜ=w�u���=@�1=��<��4=�k��:
���A_=��u=|��=k�->���g�"<8�2<]�S=@
��vy�����Z��;���p�;Ro���=w�^=F�ȼ:/>�@<�ʑ=e�=/E�="g��M�<�=H_<Ӈ������[��_�<R+y<��B�R,�<'Ɉ=B=�V�<!U�=��0�Xt�� �q��<+;��!=�7=�Q�;1�=�n=k�d=}h<w|&=�L��9�e=.�#=b!μ�:�1����b=�Y�=Ҵ#=x�=M�<s��;��=�פ��N=Mn�<�H=�*=,⠻ZA1<iv_=��=&=��=D�=}���sT���߆<���㓣<%9ܺ�G=n=ob����'<��=턼<Y3�=U���l��p�=��
�w}(<�,e�b��0���Ӽ�'=�]U�U���d�I�j=�#�I+p>y����J�'���G�6���������O��0Q=����7轄12�(G���J�=��=_e=iyʽ���B����p=pA=��y��i�|�.'n>�`!����^�J�G���8=Y	�$��;I������>w�%=Џ�OBt=�D����=Ld�����d��٣�F��������&���=�tr=��=���<B����)�;pXN=�\=B��=Zq�;�RF���b=*1�a�<Y-c=���;�MH=�(v���=�J�=�y��!��=��b=�����=�Z�=����Ǣ�;�n�<�3�=0�q=����)�w���3=�e1=�O��6
;5|����a=Pπ�Ep����=N֓=r\9>'��<lp���4�3�>w $=Hs�M��2:{��S�<D��L+�
�[��J�=ܳ=�9*>�i�=��x=oƙ���ѻg'h=D�<4�1=��h=%��=�%��Rm��'!�T<�!���Pf<%s�<��=0�=�e�=T�=�K�<<0�h�H=���=hzP=�(>\剽�:=��g=
�< {�w��=���Mw=�߼ǳ��g
��+J ��~��������<�
�<���=�p�<֥�:����,= ���W���E>�EH=(��<�J�<P�l=�ӈ=���,,	=Yۼ���<db��<FEI>�1[<čj���O=k��<�<���=���=p�e��j��G��aD��{����=�� =|�=��=�=�~����R��3���x��=�!�=�< �Y==YAh=��@=}�"=$'�=(��<�<*�=�ɜ����<���;ˊ>��=�.��˼!�W��=`�<@��<7)f=�-I<���=`G�=m���������=F%>��M>"�<�ݫ=k9�����=��;_<[t���<�q��r�~=���<�D�=${�=�>=�>>j�.���<�?�=U��=�w�=�3�<�_%�)�o���=�g[<P�<������5�������#�+p ;J�>41���H�`���&l���m=!x�.�W=qI=�.'=��$=��=�#=��/=���=��<�cS=&*Ľ;J�n԰�c=�+�=w���ͽ��<m:=2]E�jϋ���>K2�=�@h��2>�U��雽xh��>PRA=�8!�Uw=Q?�կ3��悼};��v=�3�=H�'}>���N��=4�%=G/�<�!&=��1<X���*�=�p_�5DO=��9=P<`;���Ê9=%�;���r�W;��=,yA<?����<��i=�<'�(�fC<yt�=��==��Ť��� �eC��o����<S盼���=��<�z5���<���=�y����+�2=b*ѽ�N�<�bK=P���}��f=���<Ok�<{�<�ʹ���nX#<�Ѽ+�����=��Ƽ�����~R;����z����:���<D����=t=�!=ClJ<���=l��<�e���K����%�����g��:�I�r2�=�g�Ա齜㦻ڔ���=b3�=W�.(G�u��=�_�=Y֢=������M<��m������1�=�L�<��=bl+=O��*�_=����S�I���>f�L=�/���GE�lm�<��8=����CBq��ڼD*ټ��#=�>EZ����<a�=M��=� �<��;� P=�=(&��<%f���k<�*�=E:���=ܯ!=햤=��<��w<N��<2G
EStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp��
EStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@ *
dtype0*��
value��B��@ *��u�4�j�j=|�=x���#���x=�(�xͽ���=ۚ>N�ú�D<esa;��_��4�=M�񽇍�J����5�;�'@�Ws�=���<!�?=|B���{�<6�z�[��=g�.�ƹ=�=����6���;)=��޼=�=:�Z ��q鲻N�H=|ӎ;�t����������K=\-@�)%�<6=����J��ۢ�.���f.<me@�{+�=c�$�R��:��</��=���<Be���2E�� ��&C�=a�G�c�q��,e�{�s�'��P�C�2�={��<̽��><���=��eV��&*=��ݽ^�>�����������9Pս���Z��:�%V=�!�G<�;ARV���M��B=��<N������<�:�=.4=�`л��.��={�"90�<fq�<�B~=0r�; <�=��<�ذ���q��F�L�=��	��<���K<���;��<�]�-Z�=��O=��j=&ە�	9=��<��L��b�<�v,=�w�<�x�����y�g=�x=~� WU��<
R<��" ��0X=0ks9�JM=�N)��J1����_�=h�a�)����d�`0���|/=;Rd��P�<�Ӽ(�=�;���<ڻ�.�=�J��lE�<؋<��5< �ǻ���<k؝=݁�0 9����=���=�E����漯�<�1�<'���8�:]�B<�U�;�,����=B���
;=S�<H�t��W���_�=L��<����;��<��O��7��:�=q.K<��,<�����=s��<�ז<!F������|l���Y��<
�s=����p<�
>:�m=�:3=)��r�4�檛�Vg>a	f�Р�<�bI=pa�=�f�<W�i�)5=�U<���=٪�=�V��}���;e�ʔ���;,����x������O˼^�c�S�2�J�q=�������wÿ=k��=��=|�k�);d���b�=���=�Q�<�6Ƽ�ģ<;K<���=Y7�<�G���M9w(=>�ڽ!�<_��=8^��(E;����DM½|�V��E&=�<�<e�T�Vp;�f��><.<G�|�~@�=�������<!nӽ����O���f{=��=yV���ة��@�<i�*�@��<M[M=�RO��4߽�l�<e���e��=A(<���oyX=���=YH+=oD����(==�{�$�ڼ�-�\@�=�t4=FU���9�=�~O:�=!#&�$�作�g��ț�;�<8Ck�ڳ��p��6��y%��~㌼v#�G�}��z#<f�����B�kI����<'�ƽ����ƻ���ɴ6=�i:��k�����f������=} =Z������4��z��'h�=ω׼�y�=�=Ya=ej��ի�=�l=W�=6��<㸺<r`�<�ҥ=�s��]�<"i�<�=���<B�<�{�����x����<�y=^�=i+���oE��>�������=�j������&ܐ�EDK<�4�A=¥�<���<V#�:KU��P$=M�b9����BQ��������;C����u�!=w��=I�����ƽ:��V1F=2�=r�~��p]="{�>h"k��ܻ=w�V�l����}�ᬽ{O��@U��
%�	ܽ
���6�ܙ��xɽp����f>w�+�-�>r91>vd=HQ��=�T�nN��m�>�#=l�8=�9?�1��;�a=`�m8B�Y�R�w;�,��+�P��=!�%�a��	��;T�[�-`F=�D����/;A����j�<�{�=3�Ȼw�<'�1=��=QF����=�8�=q��=B����*H�;VU���I]�
�5=�{�<�H�BL�=�U�<՘�;p���ce==�3����BD[�8� �����>�~�=sN�=]@��9<�^<Y�<1�;�y<�C>Gg�<+_\=�!�5��@>jՓ=J�b =��(�~��<�r,=�܂=R�ݽ��8���->�_=S�=r�˼H<�=��M<t�Ļս<�̷�iK=t��<e���T�N��
�Ⱥ�=F�>x��G���;��<�Y>�3�=p���8\=ݏ�<�R=Ȇ~=F�9=qǽ)��s>��<Tp��SA���2��CI���=#�l�Z $=TL"=��g��<�*q=�[�=a�	>Y�=/� <�������ċ��?׽�Z�=�U|=���=�н��ν1�黦��ʖ�<��������H�=�U0=aP`=�z�ܼQ�c�"�=h�K=�KH=�=�(A=�D���k9��*ȸ��8dl��W�=윎��	��<�OD��B`=��(=8m�= $�O璼��g��\���h=�x����<\�s��<(�ȳ�<��!=��]�#�	���^���o�RM�=pn�;�����=ڗ�=��=;8�ZG�=-|�=
�<�Α��Av��=����G�U��;n}�=��_�g�h5�n;z�Ҽg���>"�����2���=]z��I��=���h�(���<C��<�Gg�g��Ҿ>W_�<��=�����=���=^|�=`��경Å��.���K?���D�['%=e�=z&��o�%��g�=Z���#��*���I��<�U>sJ4=�J�=D�3��=V�Ž1��b�<�����IZ��\�=y�ɖ3�{>�=�@½o�=��l�~�>��`>�� �+�=
�
=�u�=�U���T���
��T���=��⽆�����<"��=g͔��G=�����9>�0mS��<.�B<af=OU�<��=gл�����S��� 4=ִ@���>}���I ��Hx=����Y�=��^=���=�ga���;
��:j[V�_�<3�d=
 ҽ2�@�ŜL>(�x<E�/=� ������MD����<>�;^��:O΀< UV=a�;�3S<�"��rEq=��:��=E�ٽ���<yd�;1Y#�}�=�	�=���=�I�9b"��ü"���/�<֦U�n�:�D'��(�=�lj<�X<�`"�s[����R�y/���r����Ѽ�cf��"=*	ؼm��<���	ٺ=ןt=�<=_��P���S��=|����=�N�<n��=(��7�=d2I�j���22�F >gA]=5��:��=�\����*�+���ʽ���x���}=���=8<<=���>�>�l*;ev/=#˽��C=H-=�Ma�]ڃ��'��8-=���<��e=k����I���t��w���(>�����ҽ�ޠ�ƨ=����G��;��w�$�޻�Ȓ����<Ԇ;�^��C==�<�Gv��W�;��<? �=�4=x����x鼁G~�4�U�!���0=�׋<������u5Ѽ�<)����V=�»���]ԇ<�k��ۙ=�]X=o�H�a
t�,�:���2���[����Ԯ�9�$�=�33�3ټ=��>������Ӽv>.�+�� P��c=U&-�x��=d��=����Eƽ����!G!��︽�g�Z=�����k���l=��<-4�=:���:\�J�}�gM%;�ab<Q��1�߽b C=ҩ�=�2=q��<9w<�A��k�x=���_�������O�?>�3�W�6;�����6|=BW���^��>�="a!<VQ=��=�3�<��;)���]��>���=�����W��=/=��=3�!�"�=nB;�����Wy=��<��Ͻ��<�{�;K�=��L��R�<����Aa�>�bs=\�{���==��<$X�=�?�=�Jo�!�[� �=T�ż�T�=��K=v,=uS་�=�x=��z=�D���h�#`��b�=2z<&�p<��ý\+�=3l>��=���=�hL����o[ѽ�%�*��<G�A�P˼<,�=�'������gÚ��"ռ($&=X�<�m���<����R倽9�r=
��=D�<�(��<�
> �<������<XT����M=�{��!�߼&MB����<o!%�J�p;}xG��L�=$�=�F��X|�<;���]{_��=9�[��R�����1;�ʡ=�F�=���=B�"���'��X���<�9ɽ����Gн��K�������<*���i�=��X=х����=�x��!]0��ֽ*��=A����%��2,>	'X��%��{3ʽտ;=��Z>�<q=��~��˼>��<Ӵw����g ]�X�%�`[u�/Ȩ<��X=lژ�.�=<"y�<R	=�JV�<�{ �O�>��?=
VH���b����=J��<�yսe~��=H��A,�7�<�Y�=�����"����=k�̼�NC=*(��8���5����ǲ<-/����׽~�*=
:ȽCrE=�};�n�;v� =' <TL�+gn�Z��={�m����=��}	�<0�&�4V'=�)λ����C����M>	G����W;�7�<f5��g����WN����n^*�Y�t=����	=�'=�d�;�
�-��<1���O=,F�<ss[=lk���V;ND�d��=�\+�U|<� �	|���Q{=�0C=���s����m�=����ð���ԼK=�����eN���@;��=���<�D�=���|�:jQ4�O�_=�
<��<q�ʽ���<
��;֜ڼ�.w=A�m�T�=y�=�x,<@ 
���D��k=���;��>8Ͻ��l�+����=Bk�=�>��@�ʦ�,�j�<bW�:A�;?v8=Q�\=U�=��=~Q�n�=ٴ���q=����׽��*=�
��k�%�M�2�$�={���Z��7��A$�=S�>K�n=�=ܽ$3�V��=N�=g�,=�j�-�t��Ӷ��T�<�=��N=����y��=��W�<Y <���=ǽ�=�>�+	=�������E�<�51��Yr=z�=ʀR=$�5�x♽<_i=��i�x�0=AG=0���*�i�=�&��C�g=urY�7Ĺ��>���-=f�u=n�=5�q���<�fۼ��Y=�U!���)<�9f;���;&���n����&=M�=w��<.�ý�gh��$ʼ�4;>��<g�������>��>��L]����3��3,>�;��\��=R� ��Ѐ���4<����J�H�z�<k9��6��v�����5�y� ����=NϪ=��t������7/=S3��ǽV�$��W(�s�����<�ֽw��B=8F>&�<e�=m�9��<�S�$�;���i��Ͻ�u��W6,�,&�<�G=B��ᓄ�zk���>ِ9�k���!&E=�z �MTռ��=� �=�.N��V6���;`��<<]û��1<�Ù�  `���>�S�<�"=�>�~OW�X�a�Ȕ���8<�����O����<�̽o�=;E�=���<57�ڪ>����U���5��/ٽk�=PC-��&q=2r�q+ݽ����y��Nk=P��\Pi�{�	�{ɯ<n��=���<U� ��?�=F`�|*��j]=r��3A9�SC=��=rnW=�5��u=	A7=�/>�L���E��2l=R�'���ν��齵�=�Vp�����A%��Z	>�)3������ڽ�8=�6f>0�=@�*=If��.yɽ�z���Ƚ`��l�-���[�&���̹lK�����T&�<l���v->��\�>��->>�����=;�:�'�=f��H�<թ�;xXo������<>6�<ʚ���~�=��4=��=����B�����V=)�O��e���<�ļV����!@�S Z��i;�3=�>Q�\Xa��-��ָ� ��=7�1<U��=��⽇�K=�����y�����1�=�+��=�5�f�!<k��=�N�
���i�ֽ	�E���\�E��9�8�<�1�BE<T?�< �w��d��1�<5^���R�����������Q[�<��>V�"='S̼d���g0˽@�ҽ{D=?�սnܧ;��0�f�=䇦=���;!�E;��E-�_#= @'<��ӼV��������=s�h=��*=D}&<�{�=��=�`��F�R�%�`=�!��K�<NZ-�I�T=ɖȽ�ǽ:e��n�j��=MoĽ �=Qˠ=�J�=�ʹ=j��=�� �9wC=F�k�<m9�=h=12T=E�=O�ནG;�=p��:�=��>R������t�2=pX���=>\>�o=��+�S���<v<O�#=�R��3R���k<%6=��=+��<٩5�ֲV<�'u�4�=�D�=K��:�#��=ݘ�=��=�V���Y��j�=��P=���� ǽFF>(�L=����i�X�ӻ>���d�&<#Lp�G$���&�=���=�O�<�Y-=����(��=X�Z=
h>'pZ�/��V4�y�\����=��z=��ƽ�O��z:�14��*�U��<'�M>NF>?U���ռJ^�=(����)��B)@<��0�G�v<Sf��x�=��<o�v=�Ò�&~����x/��a�<� <�'a��5��s˼dFʻ:��<H���&<Ũ���6��:h����ν�?F;=�z��%�/=
b=�o�D4_;���\m=�^�"�'=�H������9�8=�W�Eh�!��<*p�=ܐ#��b�;G߼4�Z;	D�;���;��;�,v=����;��=W�ż�<��=�	=D\����<����G�;� �<B�����9�d!=�S�9~*�U����Ls>��<�i�;;s=E��OC�=-�����N��ʪ<�;ս+O����e�r����Z=�t��`Zùf.�	�<�<��0<y���6�'b>�M=]k=>
)��{�=�3���q�%Q�=�s>���:��ռf��;�t���M:bni��^����� C<Z;A� \ݼ�Tg�0�<�;w�-�=xo���nK�>��=�� =�7��������`��=��ݽ��WPӺ�m�=KC�=�{��X6�=� `��y<0��=Z��[싽Y����3�(>
��=m�e=BL^;)�Z�0/ļ�e���U%�~�ż)����ü�T����!<?�(�$c�<l�>R\=�W<��O;�����p-ݼ���; ��_��<��=��= dĽ�����K=9E>m=�	H=ņ��7�G���L���MI�=��ν���;u<���S�X��<�>H�!�5>�_I�Z>�Jg�;�=/���=�*�(��=J����W���*M��N=������0�ʺA]>�H�= G=��S�^��씾��C����`����]����;I���ɿ=��:��D�;�]=���=R���㬻k�=�x��,2=-09��>�!4�A���ᠼ��J��8�<�G�<��5��_�R >�-=�c�=�4i����<�_Z�~2�=�񚻿��<�T��fH=�=���%=ɿ�<��J=���=�ֽ�&Y��&=���{�Y�|\����=s�/�Xa�W}P<bj�=�徽Km��`��R���=>��=�]�=Q��-�{��1н�$;���ս͢u����U䐽�t�<�W��wʽY">y)���>glϽ�=�>�J�������=��=�,��̋<�i	=<�<xY==�F��J=���=���<?
=K�=?픽���=1mD���޼p���âQ����=i����S��m�<ʟ�=l8=�m��D��<|B8<��'<~��<<\�_�=���=+o�<��ѽ�5z�6����0�<�(=X�g�O�3��P���=���=�y1>dO�
�o�{q�W"�<9�J<l�=R^=�#R=�e��=�-=���<���<��o=�կ=�����M[�7_=�v�g��j<��p���FO�N2�?��<���;��H=Aܽ����|��=3D�=���<��=�0�� ��m�u���E�y=�BV�D��,"��c�<��L=IX�=���=���<ٖ�=��ĩ����=�A��#>���=�k�=�����<|��#����<���=>ʨ��˽bO�<%+��Q>#��2���7���\���ٞ=R}{;.ї��=^K<R]ɼC�=�lpf<ڦZ=;l�=��E�����ʐ��#4=EQ̼��<����mU.�2�<=�M׼[D =v2,�V�=�����2����|; S�<t����~�<�)��m�%���:s�=fQ�ST�<7�C�L.=��l�n3�=�)<�<?�ؽVAX=�k�ab��M��P�i���o<)�=�ś=� %��;O�(>S�=&����P�^��>��;!��=x�=�tu�C�=�q���9;]i��r;(�=xP�=(���]YF=J���u�#=�b�=⬘=�[�#`ؽ ��=kޑ�A�	���=z	�<��Y:����Ѓ�k9=ݎV=���|��=;cӼ2)�� �Z=�ڼ��t����\����=��=���+=�s�<��ܽ�V�=�k����jki���׻��s=��ӽ4�=�Y����<X�R���>��B���3�I<t��=
�=Ȝ��Y����쿽y|�=���<퟼��A�%l����ք-��ES=
H�=D�4��"'=Ѿ�<џ�=4q�=4��<gr��½=�K��nf���="�⼿�=���<8�<;^��K�: ��<�#=󴿼���C_�=]UҼő�=ӧ9��Wûr���]�=�%�=�]ǻu�7=���;$"�4����;��^=T��;�\ʼ���?u��@4=�D�nhn�I�Žԁ�^Yt=�u=��Ǽ�Z]��B=R,�=��=B��t�B�/�>o��=�������R��<h'��!,=ï$>7`}<^M�=��Ѽ��4��ֺ=���.U>[�,>���=����z��AM�=��7�U�!=lG�!��0>=�=K�8�{�q��}-�t�n=wy�<r5<�3'=��=�+?=�;���C<RM<%�5=!"=sR<n�L������'=���-��<�l=�J����O�k�=���;��<
�<ڇN=7ۂ��5=k�o=���=�)=�=q;->t㧽X��N4<�۳���P��%��pW�=p���Ԧ���D=��=���=�x�=(�=K�ٞ�=���<��=/�8=���=[]N=�A�
U�=�*z=n$�L�Z<u��=��N=\w��S=%c�=�+_>��(��?���Z<h��=���=|lf�$n=�j�=�!N�z'�= ҟ=L4���.�<J;�<y�?=@	*=�^�=4U�=9�= o;�>	�+=�����G:�g</�=?�x���.��/���l�<�$H�F󽼪�7�u_�<B��=4��;�������_w�S�����<=YՏ��/=���<5-�=����6�;�< =��>��i�� >~_���b =T�>�S����5=	!=tN�;�6=�61=d�*<���=u(=_YA�D�=�(���H=���=%WC=�<&�=X��NF�=c�< �(�<�=��=�s�<��=oW�BM�=d{N=#��=ƫA;����<�4�<Yoͽ�>=�z<u+��ү���V�G=%4�=T6:��<Y�;�-��3g�=�YѼk����
=oT�=�Ӂ=�!�=������#�]�.�������=-�$��;��W&=�9�<^�=�B�</b0<b��&8˽��s=�9<��V=� 0��hʽ�p�=���<�k����2�^��0�=��]=�k�F��=B��=i���\�=*s*=�J���`=�ٰ����6vZ=K�n;��=��#=Ec�=�c���3��y�=���<��[��j˽��ŝ?>*��=<�:�Y�=�>x��E>��x=-��<�����Žա½�;�=�w;=@�= �vOp=�3�=�ݼ�vU�
����"�0<��>q���9�c��ϧ=M,>D���&=��_�hq�<S�g弴%2�!�м>�&�*=��=F
x�bG=8'�=D�=M��=R�<H�༾�ʼk=�Yb�}m�=���=d�-=�|;�:�!4=�aL=vuA=C��=)K�=y�ս�� �N��Ѫ=}P�=�En��7�<:�.<ח=g�n=����s��V�n=g�=g��ul�=<���^+�=v���+�w<^�=���=�F2<ȧg=S��<G�a�x:�:��$���= �=�ab�ݒؽm�:,^�=56=��_=��;ԝ$�E�T=}����q��4�S�1�me>��>�ݼ��jP��2$=�'�=��>��&=�Z�=1Gz:"�6��g��G���l<���<u�ż�9=4]��uO�����/��=W>,�5��e���7�<
g=?��=��=P�M<3������j�=YU:=�>�<4Mx=ޘ+=DӃ=�K;36=�K�=D�����=xF=�=�-{='R�=l<F�<ɝ]=o=�9�=�E�=��=�,/� ��=,!Q:�i�$�<���=��==��=�<�pk=�6G=���/��9�[<ף<�/=�Q�m��<�>q]��r%�=��5>��(=N4=xt�=
��<ұ=n]�=~��=�_v=F�r=�f�=�ҽ�zM=u�۽��>3,�=CG�=V�=.ה=�!���,=b<�=��ӽ0��<`ͽ�y���f�=)��1��G�">X��U�=�8�=�G=e<�=�=����O�=��S=��=�7W<���3��<w��;ѵQ=�vp�!0=ū�=[�<����������<@�=�a�=�X���z�9�
��=L!�<��ϼ��q����<x\ʽ�=]�=���=�]�;�ٹ=[�g�K���!�J�H���lT=�˳=��������=�|>����q4�P;�i>"i>o9�=�` >D⺽fa>U�U>�--���U��0~J>7��>�
=<*">�==��F�N�=x��=Q*A=D�<M���N�=S�S=8&��XM;I=>�7>w�Һ��}���H�w��<�2�=��=ӜŽF��=���=�e�̱�;�'��}��=�����D=���=.��w;=�E=��-=B1��HR;��=���ʒ<�>���<��>t�>��2����=����pRE<(h�<�`=$�ϻ��<)=�k�=W��=,��k�E<gXt��T�=V��=]�߼k;���~�=�=�pU=�Z�Sya=�~�=���=h�L=��H=�ф=�l�=of>ʕ>W��~� >ˍ�=ױ<E�Q��6�<�\>h�=����j"4��=�=R��=��;�������׀;��=N	��3<V��=��/���z=�=��=���Ez�>��v=��=���=��=��&=L�)���]="�'=���=̃"�)�<��=��>�1_= Q�e��<���<LR>;
��jb9�0`��T�t=K=�=��q=^����痻H �Y�R<҇�	���=�׺<"2���=p
�/I;``�=`K�=�Em�
���=���<Ce�<1����<���=�=���X�>9��=i���${O=��%��q<�+�<v!2:��9=K�=,Z��m$K<3��=�u��q��=H򼑖�<�W�=��=�j�='���ȓ=a;�<����e[��׼ ��>��=N:>="��&�X�=���<:�"�`M)����@��=�ּ=��3��nA;���=��ݽ'r�;x��<pб={R�=�@=m�.���j=�2'��e�=��>!>������X�=�)�'ot�=�=��=�~��ϼ۽�����;��=R�N���=ry�%�z=ס=�z��E�k-�=b׽��M�=�=�w>3X��eΝ;*��<��=���=U=���=�B�=]ex�na�=�K��O#��x�=XP=�{,>�1J��͜�E��=��u=4<��v�H���B!<�S�<�����H!�=j�Ľj<>g��=��<� =c�<�c���g=�/�=h�=�8>�o�<|�s=([w�Zɛ=�$�=U�d=�C	=�c�'�=U��=5���N��� =������u>ի�=�uQ=�鱼,�=4��=��?��4>����mB��᝻�ի�#mr���.=
G[=� <0YF=�}�d2e��-=r�o><es���g=�ȼ�=��[�)N�=V��T�;�7ż4󥼴�<m�;�R<%H�<��E<I�(�X��A���T���<��x�[=�ʼ����@����=�j	�+�4<��L���Ӽ"�=���n�<��W<K����A �f����st<�]�<$3c�p�=��!=�_>y�#=�e9�PO=�������=(N�<��w��4>�S�=ॊ��@5�^�[�B������!j<e��;�\�3���κ�N�=�b8>�MQ�'^a>~ݵ=y���	�J�x�3>�BR>Z�J<�y��Ƨ�1/�>!�>AN�<RG��d2j�Z	>\�>4��<��>��i�j��F=�n�%$���>=vB��B��#IP<Ȕѽ��i��j�=C>3�+�=�=�� �b!>�f���{=aO�WQz�b�%>)U =�X��^^�<ڲL=m��4d�=\Z�������v�=Xpu=���<��=P��=eMU=��G;��6��,�=���=���=j�;5�>��w;�E���"�=�ɦ�[�"=�a=��=��޼��ܽhy\�S�����=���l=��
=h�=TF�=s��=�ԑ���������K=!�>_f	=��=1},=t�0=4_�=aֻ���=���X�=�,&��}L�ˊp=��0��d���Gc������b����<�H�J�����<�H��*�<��9�J���be����<�L��/���W����K�,= ���=��_:?!�fYN=yT#=?V�<R��;~.;�Iu< r)���;��F����< �P=b>�<vM�=��<vTC�3�Z=���<W�����;s�+=�����<;Q����/�=���=@g=�#=��9=�<�ٚ=��g��Y=-�;|��=�/�=�����r��Ƽ��,<�oټj���u��x;�ݕ;���=Q^<��۽��X=%_,>-���e=���= �'���=�U����n��J�=�I5<C�=j.�=��=ĹE=h�j=&����s=�jx=��=�a(>ë�=�L˺^���o�=U�H�J��=҄�:�s[=2��<_��f���G=��=��� D�|�׼U�}="�Q=6?�=�1�Dy�<�l�`�=�>x�<]R(>p�.=�b%�0¢=S�E=���=W�<)��=�n����k<��">@K��;=���<$T�=��۽�i���6��c=��Z��	ݽar��ü4ӂ>�6��(q�<7��:��wN���,ݹ�j ��E
�㈡�T��=�'^;�Ղ��KH=��?>�R4�q�E>l#����:O>w#=c�i=o�z�\�aB�=�l{�77��~i=?
�=��=~xp>Ҥ����M1]=)���sB�=�� >g($����=��e=]��<}��;��������L`/<g�<,�=힮<8Dl�>��=���Z'
=p��=�X<�?�="��<2>pvN�+TA<�i>�6�=��n��<����Nt��Ȯ=_R�[�d<���=������	G�(=> @�==�=#�1=q��=};>cw�=l�Q��M�</�z=S�<��V>I�@=z>�<��=7�l<�>�U�=$�'��&">m�T=`�)����X����	>����g1�<l�=;e=�{^*=��=k��<V����C=L�]�H��=��C=�<�4`���==^�H=)�h��j�<Ļ�=�_=�0�xT6����=�6=ˊ!���=��|��d/>ޢ=����﮽o䥽��=��><�=R]�=0�x=vMy<&�=�Q+�f�ս��M����h�=,4��R>�*ý��}@>��=�M�w���f=H�m:֜<�sg��iȽ#� �%=�ɰ=�	���"��N�w=P�='.�<�k[<�u*���L�I���9<�N �F8�=��Y=+~�=�ȥ�d!�=�(<�VI�N��=�I�=Ke&��%Ž,>���܀�<p��=E?�=j]U��LS��TżS������=�;��ݏ��Y�����=�>�3�<wӾ������0:�K�g<��@=7V���h�;���=�2�dw=���iIA=\��=Ë.>_V�S:�tU�=E�=#kü���=��=�$Y=�>����=��=F��<l�����=;Ǽ���@G�=ŒQ��Q�;�`= �;�Mf=�T�=.U���=�%�< �����=;��=�I=��=��=�q�=�Dڽ��<M3�=!)����=���=IIo=C�:�
�<-x�=x��=�j�8�:>���������<C8x��x��袼=:=C?�=���=`�/=yA�=��='eB��<˙�=	�>'6�=��4�g��<%y���==Z�i��=�P=ڠ�=��Z����<*��=S��=<���������T��Q)�=�E���P�^oc=�H��\]=�ӊ=ir��Z�=�΅=��I�9�Q=sɑ=�ӟ=�/>�ޡ<0���qsG���1<Y�=qƶ����=A�>��G=��=0C�=�����Ͻ�F�=,�<�������<	��<�g><���~6=C>�c�<��=P����4���è���ݼ�b���Z��U4���o��u�<C3�<�Xz>��<%51���ҽ#D�/>�:RP���Գ<�f<�9��qD�<��u�j5	=�.=�=-ҏ�`kV�Zsڼ;�5��Z���h=��ǻ��8������Z=�� �lG&=�鹽犣<݆�(��v�<݃�<�sZ<��G=�=�<p=9�<&����V�<Df�<�`�6���,=V�=�<�=����6�,/>��ȡ=�\F<�X"� >������*� p���=�=5<!n�O�^�`U�=�r�=�;�v#�F�=�=l��[�=<c=��o=�(�:�	��1䓽avi�b�>��}��˽K�=��=kϷ=��u=q�Ē��d�\m�<\%�=s]ֽ�X=���=������6: ��ȼ=i��=�>�>˽Cw)�d'>\牽��; '�<��=���<'Z��bK�<�fF��I4>��f��Rב�Ԕ�=��/=�&����D���r=�$����=O�;>���=}%	>R�=!�'=�q��FY=�!==�=�ټ�u ��[�=@�[=Iv'��$(<�um�T�켖|=̎��ׇ*�̈(=���=h鮼�mR=Q
+<�o=�'S>���;��=��7>_=t+;[��Z�>�7(M�1Ѓ�Kc��ȉ�N�3=8˻:0=��=�b[>���<��=.���Sƽ\'�=j�L>۵-;���h�i=	���u7)=���=��
żǯ�sg���4=j�k�	u�=w���6��3rD=~�<*,��������p;�=��=]�:���<�rp<�^Żq!n��T����G=~A�=�b�<�>ӟ��Z��g�=�ƚ=� t�ϻ�㊽z-D��C=;xͽ�>y�=�:��]�=��(>�>�=�3�=/�=?�=��0�J�=�r#>���=;<6=�=u�h=���E�0=��<�Z=�S<(��=�!=�.��+[�=���=1=�DБ=)�;>�v=^�=����ؓ=/֊=h�)��|�=�+�=�|=��=�>Ȥ�<�>=@�X=4�)>��=���<��r=H&���=A7.>lsý�������#5>�^�=�l�<�h)=�8t�`�>�k3>Z@>2�I>#�M=x"~>vY�;��>�	��$�=���=���x���ܧ���5��>�=�zB���T��)>���=O믽����a�H=��>�lS����=�=sA�@O�=.<>�u��]��ʊ�xB�=�v=�.齆b"���@<܏���>;��=�]�<ԝ���S�=�\=w�=-*�=@9>��Y=� }=w�=�~ս�c�<���_I�<R�f=�nH=��=ˑ��/�	��	>#ŭ=�T������[�<�8=�V=aN�[�R��=0�|���=���=�<_	>dԳ=v:��;-�=^�E;��)=�Y=���=Y	�cLj��+�=o��<��&�p����F=�E�=���=������=>b<����^��=I�S���l=���<�����=+,�=�m]��(�<��}�B�=m�U=&�ȼ	���������=B��=88�=.G{��(=L�_=��=z�C��=�H�=�4=ل=H桽��=��=u�K=�C��jx�<�BX��k=��J=~{1�[����d�;����<��=��=��>�L���76=�8D=�6=*�=)a
=[;=<��.>9.E�'���&=��߻��<L-q��6=W|��׼�)D<�BT�Xq�=H=͆��wN$��w�<��2�ʨ�=l�/�ť����<�-j�B�J��<���S��9+:濄;��V<4�ں�*�����l<Ừ=ؘ�d�
��c>��ؼi9���>NW)==|};Bؤ=�<=	!<�=��<������=��p>�A���G�D���n�=�s�=W�<F曺joS���l=y�f<5��<���<r��W�%>FY1<�ܴ�o3��ܲ���=��>��q:=0<��c=e���w&�<�����{<ŵ^9yh.��C��A��0<�l;���������ҽ��I�c�=�,���N�<[Ż=�l<A��=�aG�h����O~�"� =V�P=��)>@G>k'�<���;�G�=�0k=
�J<�.>_PW<=8=}�=�y=1�ݼUIE=<̰���O=�V�=Vϼ���=�lH����=i��=A��<�J"=�;1=˜��f^�=�6����;� D��6��wݞ�8Ti�Rn�=m��:��`�g�]��>=�
>R?>�E�vt�<�[�<ӏ<�<��;=�
	=c���5�N����={������=<�Hp���!�A��<L���&���a���<m��<R�!=�Gv��ҍ9�W�=�<=fɄ�i��<A��|ф<"hO=���<�m����=����H1��6!>2��l�=��x<�S=��C=�_n=t��=�g���;>�L�U��:�T�=� ��	��=��&��ˆ=�6��d�|=�Ah�;M�X=��
=�k�=��ɽC�ͼ��%�+����=���<��1��8��
=�5����z��C�����,Q��̄<���!�3м-���V/=iZ=8h��r���2�������;�HA�/9�=��=֊<=A����L�=D��=G� ��==W�=Fr�<Q�>P�ۼ�Z��p|<��ܽ�{�=�+=��ܻg_0<be�=�N>X;=�͔�ȫ�1�<gc>~��=����!���>�u���T*�=s=	5'�)+=�'��+Cl=�=�����u.�j<�<g��=�5�<8w��r�<ᾲ�F��D)=��P<��=��'=�v�<��= <N���S����=�L���P:�� �������=�4�=��=b��=�b��Լ<��>�v�=����j��=a�%>����%�M<ȃ�=�7p=�����+a�M���>�s=�?=-<\m�=���<�>=�]<x{ҼMBk��Rj=t�彡��=�/���z�=�">�7K����==��B��=���5�`=��=�J�<L��=j�E< ,=�l�=��;wFA��?���^�?��=���=�<D/�<�j�=q��=�AX�43�=�G�ד�����=Z����p<�s`���=s�; ����=|-���/X�J8e=�
�o��:�<����$�����E�<���=�/P�?��;$���Ǐ=��H�������p��1%��܀=?�=��<=��w��=�����J+=�R�=�}�=�޽o�����J�c�<���Ѧ�=]W���=�C>߾=���=h#�V�C=w8%=��Ƽ�!��]x<9��H��;א=��;l�_=���=�K�=��};��
=��q����=��=G�O�B�B��r��p�<3e��ӎ½�->���=Z��������;R_=E=��n��腾5�$>޷�=�P5<4h>#���=�	?=K�9�9b>��6�cO�t����sO1��A<��B�wO�e��߆�%p�O�k=������=PG���@�o5�s�h޻�&�=k3������a#��
J�t�T��~Ǽ�b��<�=dg�#�;A1=\�	�@�
+s�-_��Y��G#�l`ҽB=�:'t�6 T�+9�=�0}�-<=� �<X����=�̽=�	B=��p���>�(�<��6=.�z��氼�R>���='<���������ڼ�>��y��T=?�;�`8= 6>��A�X��$c=�;�=�c���ʄڼ�3[�?�=T/V��տ<V�½��c�_���>k'��+����j=Gw�=���=V������?�M���P>�z?>Z>��8�I�K=�I/� 57�8�A�����i=���H��u����=*d��>"�\>z���6[>8L�=+I�<�%=6��=v�"�p���%
<��1�H �=��v<%"<�o>��߽�@���B������7=��=��ʻ�=�R�=�>[IT<t�=$P¼���;J��=�7<�#=W����P�������<�d"=+ư=C��=Y�N<D�(=��='v��Mܼ�=X_ <enJ���=�L�`�;i:���=�I=+X<y^=��e=�<�=���F;�N2�UI0=ő6=d����:�O���|=.i0�7L���w;���<_¢=�6<Z#~=�r<�{��P𱽳	ɼ���4.�=�~I�L���[�������x=- =��<36�0����=I��=����q���r���0��8������mʽ_$>X���5�>��<:Vϻ1=� �����=B�t�*���$��=��h=���<W�'=����M���>�=���>���<��^�g�<~��=�9=�𑺎1�=A<g��=1=�ĕ�B�z=�*=L�=yK�b0�<I!>�>�=�K���9�����=P��<V}ٽHH�_�Q>���=cㄼ!�=����!�=H��=��!�Y�e>�������c���3�ʽ6Q�n���ͽ��'�g&~�x�9�j�=� =AD<ܻ=���=�Y�=Gz=������= ��<�9��|�=܄i��C@>�=L�W�8��Ц��=5*�=>���ߎ�<�Y=/�=x�K=eڕ=��%��"˽U*=?�ʹ/���մ-=��ὝB#=�Z�*�3m�=�:%�#�S=�=���P=�(�=9�;��<���=a�=F�H<���^N��~�������=v�=_]���>}��=��<I��X��<V�m<����� 7<�:�<��2=����+�<|��� s���A<D�#<9����o׽J�0=�Y�=	�)�Cw!=�^N��&�=-��=<;��ߕѽxV��܄�=�Vļ4�=ߔ�����g�a)����;A�=;��=6ύ=����,!l�AK}��5K=�0^��r?���4������^<��"����=>_�=.�8=n��=�7.=���V�=���o��U̺�灼b�=�:�=%K�&�>V8�(XX=I4Y=������=2,]=�?>�������=,2��c�=�cν5M��~%)=�"=H�ܼ�V�=ǈ�=7k=�H�=<�=���=K�>>�3�����PC�ҊL���=YD�~��=��=+Z�=(]<=�ػA��r�*��◽`��<;Vc�@*޽M�L����=#��]�R�<t�<MI>/E��`��=	-ʼޟ���=9���+�J=O{�=־�25���j�=Edͽ���=��=\�=Y�=z�=��>]��;x-i=��&�_��=�=�\�!�|=vs��T<��'�Ï���<&�D���c<��ɼ=|�=��;r�`=��=J(��n=�3�=�����D�:ϖ=<k;ot;=mi�=���I;>ry=��=75v=T��&�"�jai�Ä)=<
<7ꜽ�"�RD�=�x+�����DM�:[O<�c=b�;BT�;c��op�:�F����ں�:�=ە�=G�I���^�\� =�	����:��{��=ͷ�=�8=�V�<|_=(��;���i��=!�<�O=��=��O��B�=�T1�ڏJ�ޝ��J*=��ͼ�3�=�s����YY�'={��;~�="#�I�4���l>��+=l7�=sO���=%&Z�(Oy�z�L�[�X�
61<vAV�����g)<T�<�=��=��v>.犽�*>H�w=�,�<h��=&����+������ ��F=�j�;(�6=[-w��UJ�
�r���=���<t��<#���3���˵���=�P�=��5�`�I=�};=�R��V�</o��c�<5��=rG��;$��O�<_�<}�ݽ�H�p��Ķt>�*�	�̽��7�L
4�h)=����K<��u<z�k=2F�=�7�=΍H��Ŗ��Q�N=��.�����q� ��u=1�Z=B[��1���2vm�	V;4>�J	�	�.>�\�=��>4X����Y����(��	#>(+
</M�5�[�u߈>z4U=���;�@>��q�<��=F����f>s�4$޽&�Ľ꒾< �6�J=B�޽ҧa�zq7��ӌ�FM��A_=5�P=�m>髳=��=m��<]�n<��<��=̃����1=��#�@=ً=�H��@�==�L���y�9$I>��|�<&e�;�� =8�]<R�)�|x�<��5��,�=��[���X=Im*�-��=2|�M$�<�<���:�X���?��J�>;�*<o�=+%�bG=kO�=���<G䭽�`
��p�;L@|��#7=p��=�2�<��<�x�=�>�/�<��=��7���:�@�=�=�.�=�?2�F{�$����W�l�E=�k=��+=ṛ=7I�=s	��%�*��|�=��!��3��${O=1c	�n��<�#�;��.����L_�<x��<?�Y=a毼��_=T��ɯ�=�UG<�$�;�K���uw<`��3��=�去XN%�Q	@��s�=� �<z�;���G|+=T����K="R���O�'-6=��=+��=��<Q���*���nd==Ѽ�L+=
�����1�7=��7�org�@('����Y-�=�� ��k�� R�Ԁ�<�@Ľڴ��
>� <xd�=��!<���<	�>ٛl�J�r�=m�==��=珔����g{�R7ѽ$`�<'��=��|=-�,=̤=�R�=�D-����=��<��e�"�i="���;J�μ�R{=�=<��`P�)>�^�=�󳼟k���d�<��>�u���=�[���=A�="ɒ<s졼|ʀ<"��:+e=S�-=8� =Y�o�]q�<>ן=�8*=����䧽b5n�v��=�j�<f��=����S�|>�����u���O=m��<��[=�:<a�=����	��|�=�ġ;��=��<�<���I>�Q��a�=�ؠ=ɑ<��!>�^=*>=��=��=Q�#=��<�J>[ߛ�~ڊ=�=�Ļ�X�=^_<��>��<l���~�)����<��z=6�*;�ϭ����7�=���=i0���K=Q	��>�a�=e�����=�.�d���o�=��N�Tu[��-�<������ڽ����(�KX4��D�=�ɨ��㦽�3�<��8<��<�#�<`�`<>�<��w;�V<�+j��W���&�=��H��(�StX=�u��-5O=(�F�"�5>c��<趝<	'o=�"�=�� ;&T��ރ�<����*����H���㫽�k�;/������4>�gQ<�5�=��8��<�M�=g�����K��н���=s��=Nj^<�ci��7&=�_f:%�ͼ23��/�=��[�]�O=Ϗ�8"��;=8�="�H��҆�｠�����B��V�=�~��6�/��=>"��=������J��7�� �C=U�<�Ԏ�[�����0>{��<�4ͽ�vh>
��>E�=JZU����=������Yؽ�O}�
e��¼`�<��M��!$��� �!���+2=����x#<AW�N���`f�=
��<���=�FN=;y<C�=�]�<|�<5�'<l&�=����H=,ȸ�ip<�4��ũ='��=C�M=N��=<R�� =i�<L�_=��=1�=�J�=�
ϼ��%=5�G��qK���=_��=���<9�D�l�F<
�m�Gf� ў=c1Q���Y=��d<[+"���½}��=ǜ�%�=�0�=Pe�<��=��=.��=�d:�+9�=h�������=m�K<n��������<N���֜�<^N����<y�=˙Ͻ"�,=��<ۆ&�Rt\�����x�=0��=�&�<��ؽd]�V��B�i��9=�Y=���=�&.<�%����O=�X��@x�<c�[<K��=��B=��=~�	�4K�=��c�Ivl�� �=��=��F<�(0�"�=x��I7�=z��=L��;���=�!H�������Խ�4D�����ſ��}7=���;G�`���=;����ͽ)e��:>Z�T=��=Ϛ.�9�����=MQü+�@�a�T=�=�Wü�.l=E׽<�>��l�W�0�=�:�=O^�<��L8FW����<�
���\�<Yr=�=� ��>�=ׇ=]������<޺����H<�׈=OQ���+�=�7��]?>oY�U���ڪ��<=O0,=�_4>>�C��-(��T�=�����<^��<��t��ɾ��P>̒>SA>*~����ڽ����~��>��<(0��7����Q����<�1��мL�Ǽy�¼}�&>�讽��>
�'=׀U<�����B��=!��C�N�W<� ��ٻW�ˇ�;V���<o<L��/�Q=JN�<*�ƻz98�~<�=H�;����W𻳟�<��̽"��:o�=c���j�<���<�n=��N*�tG�Կ�#Q <�命��<a	�=�u���I���檺��.<e�<d�"��8���<��@=H|=���<W�$�\|�<��)M���*=c�{���^<V_�=Fc='��<lLP=�_=���.O������q�<򜾼�K���U]�>�=�ֹ=���;е�=��лA��<CP)>�*=����Kܺ�����<v�>�d���+�<tNL�ʊ�=�WD9���<M:�<Ӵ�=��`=D!�<�7;7T�3�&>D9�; ��Z�<EFۼH2�<�|;��"=_K}<Yx�<Wq�;���#�^=�0=jpq��D�t>�K6�F�Y=R�M=�Z"��d><IG�=��d=�S='�=�E�	��;2�=?�V<T(�=d�ʼg@�=����{=������pq��9>� (�2
�ڰȻӜ}����=S�ʽ��'�{=߂�=�9�="��=�u(�	3Ƽ���j���N�6<�� hM=�N��Ќ�b����ռ�����0�=��,>T$����>�e=u��=_Q=�B*����դ`=c��=��>���>�=)�=���<��=�$I���>p�=0��#�:>�p�J���c>V�����14���ѽW�h�u�ɒ�0�T�&(>k���Z�&��=ݏ�=}f��0.=պv=�ԗ=ܘ���u�=kՓ��� >J>�I<'O,�������ۼ�=�=!�9�k�>���+޼s|�=y�=����;�X��Ւ� ��<}vݽe#�,�z=yWս���}F<@�=Kۤ= 5���=�W�=���<�:�<;��=(_a=\�=��q͵�hz�<:�k�no�;F�;=�Eh=�E=���<��=f��ʸg���;eM�<0c�=�V���궽�Ą�~q==��;T^���@>��%=���{B�n���(�U>{�;:$��g�c\`>)�=MxG�UY�=qԾ�� >��>zN�[>.���=#�ݽ��љ��%��齨h��[��@m�NI,��
᾵=
�o�=Վ��>`�=h�K�;�&��Ky=�9X=�i1=����I�<�@�=�

=9>�m���鼊�<�=7p�;�^:*�=J]�=aC�=;�\�}�R=���=�u��Ƽ�b��l{���c�e1v=����=��wR���y=���Ӊ���G�=�L�=��M��Ց�F�*=�;=K��<�������L<�8��ٚ<w�h<{�=!�{=E��<��>��>b	�=��&�2����=
�d��:�=�u����w=Y�p�^��c>��=�O=��ٽ���=���=�=�f��L`�Ϣ�=���+��=T٠<�m���'�٪<(X>�$�;BkE�명=@�ǽ]�@=�E;3����Ɂ��`˽��d��2���B=I��..���0�YR���o�}�s�kK�=�� ;��|��,E=$�^=��P=�	��yz���۽��<�R�w�<�c����=�6�=���<V�?>���;g�=xܕ=g�=�[�=�p�=_W�=Z���A����h���e�絹=�\:;�*;�3�<Q:���y+f��'�v��=򺇽�$	;��<���=sd"=.��<6C+�8!Y�\�ؼ�T:�F��<�)��,�����`S'=�H�<ۚ�=�����=����%���L<OPU�2G
EStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp�
FStatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:*
dtype0*�
value�B�*���L�r������IW��%��SL�x��=��_=��ս��^�1�ǽ��v�+�ڽ_>�sѽ���V����#���=�����*��g/�wֻ�k;�|��t$���:b����`��n���<p#h=��V���!={: =��T��O��/�>q�<�=�=�柽���;��W��}�==�� ��p����a�;�I�<��N�����X=����h�����b���U���8���|=�{8<>��<6�_�oP<��%�ϊ�<s��k��R���̾?=��2����V׼RY�=�Y?�A�=��7=�.]�N�༻�O=�k�=� )�qT��g��r�î�=1 <(K����x=��]����:OU>��>��!=�C�=3����=&f2=T�<��<�!8�?G=�i���6H=1$�������#��B=�O�O��97��nx>�{F��A��<-t߽��>ɇ��$�IgU<����gn<e�=���QQ�;G�5���<G�<�z�@K���V�%��;qX���袽{7������i4���u��s���d�彐�@�!��;P)���`�N�6����m���$n�ߎ>>���:��=g���ʝ��V������ �ɽ���<��"=�0����E<s����>U�Z3�	i>T����J(�������=N��<|Ц;²���2�F>���79�M�����JEC�?���ʷ[�#C<
+��������!�0c�;01�=�}r=�턽C{���ѐ�M�ҽ��*俽�H@=�����E�qS�y(����=�"�<�*="� ������ͼ��ʼ?���_�Ot���Y'�\'���ƻj���� n|<�RA��y�=JU��p��LԽ�����#��z2<�P����]�l5��T������<���=����e=Hq�;H��h6�<�ޥ��6������鮀�Zm��z.�<3f��↾-�L���8=im>�?��i�=3�j=����׫ԻB��=��^��[�=�*�=Pw��F�={Sź�o����;V�>t�R����=D.��X�=�E�<w��=U�7=ӭ�=��=�Ž��|=��߻�y�<\���7Y(=Е!=Na�="փ=�߽�v=^1��š�W�����<��O=��A��sj����1��� UB=���<a{���J�=�S=M%�=�/�=2>"��ꭽlHý7�G����<�Ȗ1�����j�<�I�=GH�*�S���S�z�H=O+�<�f=�Q#:�)��iﺬSE=��K<$Ka�n����v���eQ�F�=�������Q	�є5<2�;�bz����*s�=[^=�K���v<°�<$��<��.=c���B�x=�_����=[���o��^з=<���u�]���=��e�	Hн��-=�pg���ݽ�`k�@{����e�<�Fս��_�͘'��;��}=�c�=�����z=0�\=���=�>�;3�\'�6_��2+��/�5rK��c��b�=g�-=c������a��c�8�jH�����<Є��2D^��A�����r�ѻ��E�A�>�.�oH�����=��<d��=��S�豸=:�=\O�=�b��i�;
;}�W�w뭽�s=��>������9���PU>4m뽲�{=�q���r=�sB=*?��-F�!�[<�\�<��޽�N��.�<���=0p>��O<�eq�L�<��=Nn�=��=�g�<%`U<d��9���(�<Z��e�ҽDK�=���\8�=z%������Y��<��4��n����F=^m<�o;��3.���s���
<���4��q�a��(��2ĻBM<�(�=�^�J��;��,=ro����=��T=�ͽ�=���;�M=�ѕ<�@Q=C�Ὃ}��c>�F��=�_���2>]	1>n�����p��<����W�����;&��'���7 �Z�Y�!��s���y=� �<yL=Wt7>ɳ	=�+ؼ�,��G�!�-=6���3=�x�a�(�T�������C�E=���$�L�0��=���=Q�=���&-��i��=dj�=g#��)��=;8�=�	�� h<�ӽ�Y�=�j< k<��=>��; <��=̘�xn�=��;.M4��5J�;?5��Kt�>@�������p�K$=t�~=H�=>m�<�ʈ=��=�>̟�<"G�<�]��Q����j�= �<(�P<�IR==���x���u��s���3ߚ�Y?�<��>��E:pz�=�I#�&��>x�	����=�� R�=�6��H�E��q�>"�nܽ��n=�aP�=�o��Ձ�<�E�\,L<v�H<&��<biy�H�,<5w=��=\Y<F�����<���>׼���،>�=�e��<���`=p�<�a'=ݘ�<������/=��۽�j���T	�G��=�4h�
�>)��=a�<3]��H=�vd=�eD<�s�<�=B�;�6�w����/j=cݼ��>�Kp��*E�=�xս�=��߽����/�=<�^���s=(?�_�G��l��mm�9�D=�8�;�����Z��󮔽���=�=�ޓ�_�'=�J
���<��.<�C<�i�= 5�<h؉��q<����&>�J��/��=�R,=&��ԏٽ�K�Y��R�Ѽ����l�^Un=0ټ�3��;�98ca��a:�Xv�9�>흄=�߽&I���=�<G�]E�=q(���&�:�/=��o�8�ؼ�YG�����*tڼ:M���V<�y=;D�=$�#����<�ۃ�{�J<y���=1|=��H�I2�=�c��鬍=h6�=x)�|��1S"�a���%�ɼ*�}=.���==�K<��<m��R}�ސ�<dQ�/2`=��f<���<1z�-c�=�S� &=����=��V��:룽�~��~�Z=�X�vr��B��=���Q���h<ߟ/>��(>��S�S�����=������%9̼�`�<��x�y��j1 �w�|12�Yv�ʯ��2g8=2�1=Z/<,5��j���=��'��㼼\x	=z�Z�%�u�ø��'�D�vp���G�;�>2H
FStatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp�a
EStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:  *
dtype0*�`
value�`B�`  *�`�ֽ
դ��->��/=Qb �%w=A 2�@7��_m�<�%мN�3����P=;��<������/<'[��z%P=GS�>�CY�-��
�C�6؛��kQ=�lӽ�	�}�=	�#�-��NK�xzY��R�<��Q=��˽�+�=�=����d2>�_;|4�<ᱍ�B}�=��<�=�=�C9=�a=nC�<lC���> ������=�iM=6G�[���W�:Ge���<m����>��;��=[@0�s�;����7�=	x"=�/��>�Gl=%�ս�n�<��v�m��=Lk�<E���rܽ�ׁ���,=y�>���<�V���*>H6 ���
��Q8���i<GC�<Ŵd�`��=:W���/��/_��o�=��<v%��S�<�r8=;��؃��y=��%=���
=��=p�F:��=6���9�2;j���X��=���=�,=|�<�ܼ��w=��^��Q=�j�=6x�<�:>�阽x���3��<�|�=�u��=l�ͺ��6�m���p1�K2�p��;^P�z4���%��;F.=���M�'�b\�ϛa���F=��W�T�2��~�_�˽�X=yFi>�Z��K?��R\��p=�q>�}!;s-�����Pf=��)���7��Њ�V�i���ټ��ff�<v =8�� ��=��ʽ��O=1�8����;���=�gU�-�a=��=�1'<νѽfe���=��E�>�=�E >�K
����={0==��$<߫
�ud=đǻ&�N�&�q�@�½菜���=B�=4�<AD:�S����w�<��p�>�P��<��m�ԛ�=g���6�<��<$`��T�I�Q���Լ�
��p3=j��c(C����=Rh��C4 =�k���p<�?=�zO�w���	����PF=?5�� w4�ۇ�=�t;�楽�w�=�>�p�C<"���X;���)8�C�L��^�=���=`h=e���>᷽��v>C�j��do���->\\I=�Z.>q:���E���{=\(�=V�b,��˭E�L�{�h�A�g�.��>�	�={�Z�:�=tW�̱=l 7���=�A<��S��~= s�=za�=O�Ƚ�f<4+ٽ��=�j�=
�"=#DZ=1��G,�=zK=���F�ռl��=��V4��k(��ݿ��B����І<��[�ɻ����7��I8&=|]�����;�S^;0(����t��%�=�I�T���w������u�=ޖ0����þ�*�=�~s�<y���##��)<�&+��K�=[��:a /��AȽv���5@����=�E�=����Y�=&�4���=v��<#·�T��=QB�pO�}J�=�����=Gk�=V�R�J� >/V�;�����Q�u�j�1��<�R��\�kID=zB�=�r}�M���Ҍ��@#=�G��S��P̩=c�d�-�g��?=2T�<�=��=����X	j=�zn��t�=W-=����3Y%=�ֻ�x���弟�@<�R<��.��[Z=k{�<0W�;�G=0ƻ��=k���l�:}�Q=ufI����=gp�<���=6"��w�r�1��=�(>Uz>��M��,x=����.�ӼMJ*�ٽ�=����?�<��~=����>x�1[� &�=!��\T8>C��=5�6=f���=��u�L��:�;��<F]�J$����=���=��۽�ߌ=��p���;�|<�.��Q�bn˼��=��>*M=�
$�����V<�/->/q@��9-��]�=2Mڻ���=��������*�<'=t�ֽ4� �1�����7�g�>	e��A�=�2�Pk�в�;!5->�@����8=�ۛ��%=�Po<�Xe=��>���< ��m�K=�0��ƽ�(��Е�Az<<�?���R;><�x=*Gk��Cݽ�>:�E�$>����1���G\�K~���S�ǿۼE.�;�	��d�=��w�]��:�0P=��&���#;����2*��.
>��5�{�*�f�x�@*���B�>=C�O��<��Y��?�=Gԏ�5����;�Ƭ�0�N��=��c�W�M������f���=Dbw=�e��V��=�"��]=^4/=��4[�<��ּ5Hr=]�-��<����������6��)>�D����C=�S�=��g�0�>p�����A�:�~:��=ȝ���`��`��Ͱ*=a�`�p��Nt��>aǊ�2
v>mU�Iw=Oo��w���6�<�tƼ�o�=�@�=Q����6=�e��==�va>���;T��`rt��h=Mۅ���({I�Z	>�:�����7d=�]��#�ڽ665=�*�-��=G��=�ܽX��=�츼2��=.Ļ�^ɼ,�f�8A�	k�=�f�=�М<:	�<Uރ�����r=
��;<����{�<=r���`F@=z�9��E��Ϭ�5�ýG���S�Ž�1�=���&i��VL>%)+=���O�=�����<���< �<=ۼ7.����=Y��=��˻�Q�X�1�$�G���,>�s���<,�X����=�Ļ���qF�~�=�#�=��_����;::�=җ��ƞ��Bb�>��=�/<m����=X$�eB�<�bO:�䲼���<H"3�4��:�	h=Q0�<���)=��<��W��3.<V��=H�K=����S��;��U=Ф<�uܚ=�Ż=&4<��ݽ����,a=k�<��E�ı>��h=�Z��,�=�u���=GD�;�.��
v<N��|�<�<� ����<Q_=gD[�9�V���ν���vȁ<,̄<C?=�����Yy	�Eb �4�=��/;	U��4��=m쵼���&k�=!��=�C���=���w?�;Z4&��xl�����8��%=�^=ӏ�=�֐��X��V=�u��M<�ܴ<��̽S9�=[��x�^���<��=z���`�<�R��Z�<2�=�^�� ����\�=hdx�昙=j�<g]�������rm=�}X��C4��;����;ŏ�<h���\��{e����=�e���<�l@=V̬���.���ݼ�¼����0�TO����[��]=�^=�\����N�>�1C>����>�0=J��<�)���)=׫���=g�.=;��=n�=p�O ��}8<�����}����/=�>}�'�������P�=����� Z��&�<g$�,�罁)�	Լ=M�D=���'���['�<�j*��Q�=�\���=rd!�FG���� =���{�
><�R=ǹ�^���� �~�!=Z��=���:֯=.��=|����Ҁ=UO��c!������LX=�қ��.�;5(#�<(�<�EW=�F��i��=B~��C���A=Ӌ;C	��f>^;5�ƼQ2:T�*��B�=��w=�ҽ��<J5�IB�P��=v�-��	�}��<.4�=���=E��<����:<Fx�=sѽy����>�o�<��V��G%���F=h�=�7��ʘ=w/����=] =�"���I�Z?2�ߧP>5#�<ȅ�=���Ƶ켼t�yӶ<��"�8"��$�=uj�<:�>WBк���ʼ��Y=�d}������o����<�v�<)|=q׎= ����=O7=G=s~ƽI�E�Ȼ����ļ =U��=�h=��ڽ*Ǡ�I[ ����=�*?��ǧ�p	t=H��5�>D�̽����[J=Y�R;m[
�	|�	g���kY���.��R�����H=q�.���=��D�D<��H=�W��ю;���W���B6�=I��=��;)�����<Y�2>·���=:�ͽ�a����=���<�6L��>��|<5��u1b��ڥ�ʤ=*�@>
�>R2����0�=e���x>�]!�#��<$f��$~�= _=����#������]�����F��ؘ�2��y���RԻ�f[��<<@���Ǆ>�_3>!�;.=����=�`2<�u�=}>=��j=%ڽ�l>|��餼]�=dہ>������Q��x�	��&�;%߂=K��=%b2�I?޽���<K� �U�\>cY��n����>��<~��=f�f=����Z��P<����@���a*�=Y､��m<>����m�=ѷ�=�3����;
H�<S�=X:!=�=|��?=�Ê�/Ď�*x&�]���2c���D�����ND�=Y(����μӾB����=v)�=-��=9�M=ք3>{�a��,��2�𻁴P=>u׻L�>�b>�p�=��J=䚞=�kb�7�<Q�#�;�$<���; ��=���=��B=[5���eC��Cu=���=�h��Y+��H��>k�6k=��+�.��=k����f<��6�� ���9=:��;;D�=2G=� ]<�a����=e�f:�=�_ż�lL=╧=5���Ai�=V$=W��=�5r<H����:�يI=胒��:������뼩�|�! x=S�U�Ń-=./�=��<�t�� 'ý��=�<
�Z�C�w��=��>��뽕��<o1ƽuD�s.� ���V�ܽ�,I��=+�d=k�C=x�'�ܙ��µ���'�=u�e�;.o�g����=� i=�C >b'4�\ж9�6�=�dN�w��;
J��ژ>�6�<����3,m�-C�=���<���=�.�L��<,�5�*������<Jݼ�k=��i�
��=�����K=�[#����U隽��c5��)o��>0���=�0�<�2�=«�;�U�=��Ӻѩ����=W�t=�=+�ؽd��A����m���>�b<��<��=��=u���.λ���J&`='|{�����z�?����+�F�^<�gw��ͻ\���=g 5<E=��k=��+=N[�<��o���̼wo�:V�5��=�l�i9<7~.<*+a�݉=#iI�MЋ=6�νf��G4U<x����R=g�T=U��<-�W��:�2��ÅA�(�=/�����$����<x���i����X�<?:�� ��<s����>�� =��}�2ɽ�*S=dYT�4Y�<{��<�C�<�����=�iL>RS༧<N=�`�;�W��SD==����7=�{꽝���i&�=�sV���s<؂���,=3A:�'���='F=�Ħ��8���A�<vN���bw��J��#G�=�W�tu}=A3����;�h�;~�</�=US�9�t�=��;�@=��l�<�>�=*U>nX��V �#�˽��0�kK��1\H�ß׽���=�i[=��X��,��������a�o>4V�=�m#���8���>����q�5>͉�Ѽm��5m>j��<��]�{�.��)ɽ�W< َ=����bٺ���m�<���<�7��j��=�r�=�*�<�-���7����=#�����P�1�����=i��=i��zG�=b�=!���av�=C�[=�k�=8Y��	n�=n��`%�;�׼_���c�z<��=��ټ����5���3�<=G�<���Ï$�!+e=4�==�Ę=�|��D�	>,�=�7��(=Ng�==̎=JJd=�B�>�%=1�Q�m&^�v�l=}�Խ��ڼ�g�3]�=�Y��F���U���C�@n<�i<��=���< 6ݼj��'=45��wgټ�m�=�q=��K�^<�ɽD�;b��=�U��
�X��6���?ʻ\�ӽ��V=�2��f���MMֽa�<�����+<�8UGȽ�a�8G󽬭4=Ѐ;>�t=����z��;�A<���=����ѽ����宼���3K����4<�Q\��+нﶣ=Bf�=&u@<1��Hή<f�>^]�1 ��QF�kμ�<�2�G�>=̿�<�fL=uߝ=� �=��ս��s����92 =����D�=��<+�<�7F=��+<9��aM�f=��A��8���z<�%:�Qg���=�w���Ʉ;�ߑ<0�A�!��6~�U=�i�=�*��v�����<��=]H�M#,�s�=������=�<�ӽ�sE=C&���������_�=�@vZ�f� =�0�g>�	=�qQ=�=<܀�<8ӡ�j��=�}�=o��r����_�<<�@��20=��:�����=h\�>Ԫ=FI[��F�<୤���<��2�1��=t	ʼ��=Ⓓ�D��몡�-CǼ>0 �׊^=�Z�=<E=f��;ɝ*��t�<�Y=�j�����<�;�(� �~�GQ=[u��<��=U��LZǽ�ǈ��+N<�W� �'=[U���=cr���Q�=��=�͡��6@��`�����<3I=3��=Bw ��`�=t�=Fa=)�<��[��ỽ��h=vo=u�׺� C�"��(�}/�=xh����>�׾�9�O{�=�0=�~c�sN��=&Sm=ӏ�=PT<s�=��U=����Π=�d3��n���d�=%w�=5��;;L�<F��=+��=U����<�S-�o)��Qx�=���<x�=]�<�a>�F	=����'���1=W7����<Z<��pt�=n�	�"=�=����&�;��ý��<F��;�B�<��;4c=⽘�&=��u�==s8��.=S�=�)��=�$�<'o��/��;�S�=͉�+~��<�!�=zo���0�<�����`�=߱#���6=���=>R�<�o;g��#(�=��=�@c�lL�<��=ϩ�� .�=%P>�=j=T��<-j>a%�<�Η<�P�=���=$�1=XB�{Z�=���������R�>Z񗽬cM=fF���l=���λ=�,=�U�=�����,�:�}=�U�����%�s=Č�<2���ͻ�=U�b�q5������ �I=�t>�fƽ�`�="=\�=�,^�w����8�<��ʺGѼ�θ=XȽs�ϼ�+B�V�<��+]�=mM�:Z�&=q^�<�ѽ���=YxӼ�_۽�%j=k�=_s�;�O�=�VŽ�6��B�;_܆<9ż����Gv�	��=��X�@�T<��h�$�=xyV��`e=����=��.ܩ�J_�=a�~�c�=Mc~=:�?=�~��j=����Ľ�	��� ����=�����<;����C׼d��<�c=���<��;���7=��=�o,����=Hav�tZ�;�Ke�M4�=i��D�c=L��=̘ =�jS�+ߠ��[�=��!�=1.�=d���/�A�r���=��,>������8��<�:�<��m�z�W�f���u���N��=�u���=�����=v��;{�?��ݠ�Wa=lOH=/�ɻz,����C<=i=���=.q=K��<�l�;����o�#=�+��@��j��(l�=nɼ)��=W�<=��Ȼ �<�>�`�;�NG���&>}�<r =�<��=�E�;����܆�`��Ԗ���t&:��1�P@l=�ˑ�i�=��=�៻NQ�<�񽻆0;�^��$��'ю=��>���gi<�D���;켽s����O;�5��
ȼ�̃=�ʮ=����&�;��<�~��>λ�J��9=/z6�@���Zg���=WPm�"A�=���<����Eռ��$�Mu��^b���佽r�=�h=��d��6�<}��2eW=6}��n��.���Q�»�6�=��]A<A�!�N=7�S<��<�3����+=k�˽�ZP<w���=����ͽ_��= q�<�g�R��a߽�M=D$�,�Ž�E�<x��=g����	�=�R\����7�=|f�s*��Ğ<�.�}|P��E�< ����ǆ;�2�=h��Z�>fa��n�eK�<�|�<}����=3�ּp�e�>�/�φ�_O弓�k=i��<�����h<u�=:�ڹ�}/=EHY=W���w��=�H�=�t<%=p���r��=8Y�I��=� z�kN�/?�=@C=����М�uv<��=�U�<u(¼�H�=uʴ=�����~{<i)>�`��������:C�9=+4���=ыk����;�h(�Ľ,zH��TٻQ狼�r��u�n=���<�+Z�������ʻ�
�=���<���=q᜼C��=����:=�.��+���,>1y���3ϻ(��CH�/1"=B���1����#�q�=��������~�\	��I2`=�tB�w�׼���ޤ���1�>�u=hn={_����&����������.��nT����=Zuw<��x��~=�L�c�=�`>ų��M���y�=�bh�t��=N>��Tc=���=ִ�=�Ҽ��s����;ܨ�=�wὯ^�=;桺V��!�#�R9�=�>Ww=�W,=
�=�K5>v��<V�ǽh�>�X��",>I	��F�m,�=��>I��=Ml���<}L�<����<ZL�<5g~��8����=O��=�=�8;�2c=��꽻�Z��Ȑ����<tx�=�h����1�H��=5�=/���
+=3pi��=^��L������|�=���;vo�O�=�P�=?(���=�m_=�&�c�&��7u=��༩��=��>]=���<K=�v�=������ʩ= �<l�f�6�q��=���<{�;�z������M�犑<�~K=�<A��G�u;7��=t�<8�~=�����>_�?�yE�=3/��	<)~s=�<T>�=��T;��=�s�S��&������=8����f<��"w=�5�<�y�=U��lc=���V�ɽk:�=!�=�V�<S�=���<�&>����?��=�y��P�</�<&��=Z�����j����u=�="O����=Y�J���<TSa���?>,6B����=ӱ=<��j>"�P�����=�$�+4�=�
>Z/�=��=4}N��0Ƚ��>>s/�����=��нA�,�y��=uX�=�C<T�F��)P����������=�b+���s��L.�j�u=�j=�iI<��F=N�<�H�=Y�����5�8<�b�Ғ=V�Q=��+��B?=h�>=/d�ʦ�P8���=ߛp�u�?����=�S�[�$<M0㼜❼�L�ݢ��p��=ePm�p�;�3���y��Gƽ��ǽB�96��=iY<4�}<1j�;Z;�
@��޽"�=*ש=��<I�=}h=�n�����=h�f=�����7<ѢW=�ڝ;�3T�������=��=���<�eQ��D��/i=���<`F�� �=����=�U�=����	���Ai�����,4��=�;=�~�<��ܽe�=U�=���=,�c=��>G�>P�_��9�#>���<RE<=�����C�=����̄=��~v���ܼ���<5��<���C�P�?{=�a��kT��^�<�O�=y}=�)ȼ��Q=%�M> ��=���=�2�TN>C���]8<J��ZݼD��=����#=k��ŦB������h��u�<a
�<�a>/=go��瞈>Na��x>CC�=��Q>m�-���O��:t�h��=m.=�`潕��=��=�p=E&н4�R=�����D=�!�<���++�=��=����M��-~=�,�=��U�}�%>Q���qӽ3����u�;��>=��<��;�k=ڬb>X諾I��;�71=�L��}0�=��udX<f����U��g;2ԭ��0��*�R���o��<�&�"&��B<%9i;!��=��3�UTS<��=���<"�j=֦��������݁]�^8 �r�Ǽ���fA.���2_ۻ���;t�x�!<�*�=�'�si�<�z�=��7�3T>���`������K!=�O���S�̷�=��L��G��髶�q4���7=M#(��F:=k_�=7ヾ�R=����L �JLP�9O>V�开b�=l�W�A��u���C<u��<y:׼}ɂ=B��=���[	<E'<=2�*����<��=��R��d<+�e=FdB=���[�>AR7=(�߽�֏=�VJ��{��!%ʽw�=�T?��<~=���<Q^=9膽p>��A�=�d�;rn+�VD��s�;_蒽b��=D1��o`��a�;;b�=1Y=�=7�r=[;�="��+�½ۼ==�*�hw�=���N��={r1�� ��;��ȼ�M��>}#ڼ�Z8������L���9�g >&�>���L��<B��<m�Ͻ���= �� *=>b�='��<'�ӽ�۽S��<j)<�-�=n3Լ�pL���(;f�=��>Kk޼{�>#��=(�d>R�七� ��=e&��N�+=̽2�=��a<��6��⦼�>q�@�A	=i4�~�F=M_�=�$�=��;w<=6����=&�ü���<Pꟼa�@ѩ��l�<IWR=y[8=���/�=��=��:�5 =w.w<��i�)Z�=�ac=Ҁ�>y� >�DýR :{U�>��i����=�=ȽG�=! <��=`i�=e�������<@ c��b�=F�A=���ӕ�<:z�;�K�=� �<�>��6>��i>Y=3��<@[�=�㰽�+�=IJ�.&0=)�=S��<"�%=H�=x�=��=�`�=g���C�����<3n����>=ǛO=�x�=d����y�=�3������	Q��C�<@��<�i^�U�F���=��=<3Q��+Ļ�>�����=�=�Pr=8��=8�=���=P�N=p$}�ͳ�<�H=��<�W�=<fO=�#��ؼ<���;��8�\H�6�=m�	��N~�C�<�R�~=ߚĽ[#��y6>u�%��]!�B4�,�<hԽ��=�+���=T��=�>]��<d�<�=�<ڻA=���<�yɼ6C�;p�:vy�<<`�;s��=���=����x�=�I�<�F��X���[;���=C�������<E����=l=�*�����=W�}��ۘ;�)=���=�bx�
e=h��=&@G="�B�f\�<�s>�=�`�v�=޵½�۫=�l�$�4<,ռ�Oν� ���ϝ=��=�t�=��+�t�=p��<�̻����B�>	=��qC�<�{�=�D�=$��<�%N=;q=�h�<ye�<k��=(k=���U<�=�Z�9{Y�����=��<)��7��xյ;i��}���_1<�Y3=�� =\��:	���N�=P=�m=��=_OD������l�=ff*>eb�<�aW:@_�<^��<�`<r��=��q<ίO=����SS,�I�^��s;oy�<��X<G�-=�Q�<;��"7w���r��s�t�����+<$�=�OS����<�ۼ57E�v�=m�=���=<�&=G3��6��=���=.�u�A�<���=��7>&�<�ˍ��:<�]=
ȋ=����K~��/�<�dr=ڽ�=���<�ʐ��?\�:`=	Ψ=(�޼d.8;�TM��<|м��;~$>6ށ�W��=��$=���Y�d<�=��>=5��P�=���<%Y�=-ͼ�_�<�=�O�=[�Ž�,��-��7�:>�%n�sU=gI�����i��Tܼ併=,!L<9�n��=����<+��4�K=�M�<����z�<���<���=/p�=UQo=���<��<�nf=Z�>������N������=I�~�|��;'~=���={Qs=
�>H�������񽂏�<����!����I��G=ֹ�;o�U=�~-�q��=m�{<\�~�`>`` �������=0b=y�=�D�=A�=@��<��b~�=�zc=�,�;,�<�:>.�<�5o���X��c�����֯�����3����>�e/��7�=��׻;:��	>+>ؤ=������=�/>�g�<����=�;�=��>%��=��漟R2�p]�=�v)>�d<���=et����=lo����>1
��t�����)�(��=�ٽ�(���v�1��=�E��iy>u�=�0�=s��<2/L�n=i�a=��P�ϵֺ򣶽�W�=��h���=�p��Ͻ\��=�AK=aƌ��Wؽ�/�4)�=������=�D���Y/�	�v=g���M�	>�F���mr=��;�S>�(��:�w�M��T��ڲ=���<Dy��	a��q%>�"|:��O�*C=C����(q�����a*н �=�Y<.7=�|>�~=-�G&�Э��q�>($#<4j���7����L�=�\�u�5�b������=���>^g;^�����=��5=�1����=u�����=�G*�{t=�M�����=���<�̽X=j�<��=xϴ��"=��;��ڽ3�|���="Ľ	�\�۽2����<��=���<�J<#Y =�r���<2G
EStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp�
 StatefulPartitionedCall/IdentityIdentity6StatefulPartitionedCall/mnist/output/Softmax:softmax:0:^StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp;^StatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOpG^StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp<^StatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOp;^StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2"
 StatefulPartitionedCall/Identity�
'Func/StatefulPartitionedCall/output/_15Identity)StatefulPartitionedCall/Identity:output:0*
T0*'
_output_shapes
:���������
2)
'Func/StatefulPartitionedCall/output/_15�
4Func/StatefulPartitionedCall/output_control_node/_16NoOp:^StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp;^StatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOpG^StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp<^StatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOp;^StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp*
_output_shapes
 26
4Func/StatefulPartitionedCall/output_control_node/_16�
IdentityIdentity0Func/StatefulPartitionedCall/output/_15:output:05^Func/StatefulPartitionedCall/output_control_node/_16*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:5 1
/
_output_shapes
:���������
��
�

"__inference__wrapped_model_2776216	
input:
6mnist_fc_1_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_1_biasadd_readvariableop_resource:
6mnist_fc_2_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_2_biasadd_readvariableop_resource:
6mnist_fc_5_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_5_biasadd_readvariableop_resource:
6mnist_fc_6_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_6_biasadd_readvariableop_resource:
6mnist_fc_9_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_9_biasadd_readvariableop_resource;
7mnist_fc_10_conv1d_expanddims_1_readvariableop_resource/
+mnist_fc_10_biasadd_readvariableop_resource/
+mnist_output_matmul_readvariableop_resource0
,mnist_output_biasadd_readvariableop_resource
identity��!mnist/fc_1/BiasAdd/ReadVariableOp�-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp�"mnist/fc_10/BiasAdd/ReadVariableOp�.mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_2/BiasAdd/ReadVariableOp�-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_5/BiasAdd/ReadVariableOp�-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_6/BiasAdd/ReadVariableOp�-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_9/BiasAdd/ReadVariableOp�-mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp�#mnist/output/BiasAdd/ReadVariableOp�"mnist/output/MatMul/ReadVariableOp{
mnist/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
mnist/flatten/Const�
mnist/flatten/ReshapeReshapeinputmnist/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
mnist/flatten/Reshape�
%mnist/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%mnist/tf.expand_dims_3/ExpandDims/dim�
!mnist/tf.expand_dims_3/ExpandDims
ExpandDimsmnist/flatten/Reshape:output:0.mnist/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2#
!mnist/tf.expand_dims_3/ExpandDims�
 mnist/fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_1/conv1d/ExpandDims/dim�
mnist/fc_1/conv1d/ExpandDims
ExpandDims*mnist/tf.expand_dims_3/ExpandDims:output:0)mnist/fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
mnist/fc_1/conv1d/ExpandDims�
-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp�
"mnist/fc_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mnist/fc_1/conv1d/ExpandDims_1/dim�
mnist/fc_1/conv1d/ExpandDims_1
ExpandDims5mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp:value:0+mnist/fc_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2 
mnist/fc_1/conv1d/ExpandDims_1�
mnist/fc_1/conv1dConv2D%mnist/fc_1/conv1d/ExpandDims:output:0'mnist/fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
mnist/fc_1/conv1d�
mnist/fc_1/conv1d/SqueezeSqueezemnist/fc_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
mnist/fc_1/conv1d/Squeeze�
!mnist/fc_1/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!mnist/fc_1/BiasAdd/ReadVariableOp�
mnist/fc_1/BiasAddBiasAdd"mnist/fc_1/conv1d/Squeeze:output:0)mnist/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
mnist/fc_1/BiasAdd~
mnist/fc_1/ReluRelumnist/fc_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
mnist/fc_1/Relu�
 mnist/fc_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_2/conv1d/ExpandDims/dim�
mnist/fc_2/conv1d/ExpandDims
ExpandDimsmnist/fc_1/Relu:activations:0)mnist/fc_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
mnist/fc_2/conv1d/ExpandDims�
-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp�
"mnist/fc_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mnist/fc_2/conv1d/ExpandDims_1/dim�
mnist/fc_2/conv1d/ExpandDims_1
ExpandDims5mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp:value:0+mnist/fc_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2 
mnist/fc_2/conv1d/ExpandDims_1�
mnist/fc_2/conv1dConv2D%mnist/fc_2/conv1d/ExpandDims:output:0'mnist/fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
mnist/fc_2/conv1d�
mnist/fc_2/conv1d/SqueezeSqueezemnist/fc_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
mnist/fc_2/conv1d/Squeeze�
!mnist/fc_2/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!mnist/fc_2/BiasAdd/ReadVariableOp�
mnist/fc_2/BiasAddBiasAdd"mnist/fc_2/conv1d/Squeeze:output:0)mnist/fc_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
mnist/fc_2/BiasAdd~
mnist/fc_2/ReluRelumnist/fc_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
mnist/fc_2/Relu�
mnist/fc_3/IdentityIdentitymnist/fc_2/Relu:activations:0*
T0*,
_output_shapes
:����������@2
mnist/fc_3/Identityx
mnist/fc_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mnist/fc_4/ExpandDims/dim�
mnist/fc_4/ExpandDims
ExpandDimsmnist/fc_3/Identity:output:0"mnist/fc_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
mnist/fc_4/ExpandDims�
mnist/fc_4/MaxPoolMaxPoolmnist/fc_4/ExpandDims:output:0*0
_output_shapes
:����������@*
ksize
*
paddingVALID*
strides
2
mnist/fc_4/MaxPool�
mnist/fc_4/SqueezeSqueezemnist/fc_4/MaxPool:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2
mnist/fc_4/Squeeze�
 mnist/fc_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_5/conv1d/ExpandDims/dim�
mnist/fc_5/conv1d/ExpandDims
ExpandDimsmnist/fc_4/Squeeze:output:0)mnist/fc_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
mnist/fc_5/conv1d/ExpandDims�
-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp�
"mnist/fc_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mnist/fc_5/conv1d/ExpandDims_1/dim�
mnist/fc_5/conv1d/ExpandDims_1
ExpandDims5mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp:value:0+mnist/fc_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2 
mnist/fc_5/conv1d/ExpandDims_1�
mnist/fc_5/conv1dConv2D%mnist/fc_5/conv1d/ExpandDims:output:0'mnist/fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
mnist/fc_5/conv1d�
mnist/fc_5/conv1d/SqueezeSqueezemnist/fc_5/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
mnist/fc_5/conv1d/Squeeze�
!mnist/fc_5/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!mnist/fc_5/BiasAdd/ReadVariableOp�
mnist/fc_5/BiasAddBiasAdd"mnist/fc_5/conv1d/Squeeze:output:0)mnist/fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_5/BiasAdd~
mnist/fc_5/ReluRelumnist/fc_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_5/Relu�
 mnist/fc_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_6/conv1d/ExpandDims/dim�
mnist/fc_6/conv1d/ExpandDims
ExpandDimsmnist/fc_5/Relu:activations:0)mnist/fc_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
mnist/fc_6/conv1d/ExpandDims�
-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp�
"mnist/fc_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mnist/fc_6/conv1d/ExpandDims_1/dim�
mnist/fc_6/conv1d/ExpandDims_1
ExpandDims5mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp:value:0+mnist/fc_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2 
mnist/fc_6/conv1d/ExpandDims_1�
mnist/fc_6/conv1dConv2D%mnist/fc_6/conv1d/ExpandDims:output:0'mnist/fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
mnist/fc_6/conv1d�
mnist/fc_6/conv1d/SqueezeSqueezemnist/fc_6/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
mnist/fc_6/conv1d/Squeeze�
!mnist/fc_6/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!mnist/fc_6/BiasAdd/ReadVariableOp�
mnist/fc_6/BiasAddBiasAdd"mnist/fc_6/conv1d/Squeeze:output:0)mnist/fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_6/BiasAdd~
mnist/fc_6/ReluRelumnist/fc_6/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_6/Relu�
mnist/fc_7/IdentityIdentitymnist/fc_6/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_7/Identityx
mnist/fc_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mnist/fc_8/ExpandDims/dim�
mnist/fc_8/ExpandDims
ExpandDimsmnist/fc_7/Identity:output:0"mnist/fc_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
mnist/fc_8/ExpandDims�
mnist/fc_8/MaxPoolMaxPoolmnist/fc_8/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingVALID*
strides
2
mnist/fc_8/MaxPool�
mnist/fc_8/SqueezeSqueezemnist/fc_8/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2
mnist/fc_8/Squeeze�
 mnist/fc_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_9/conv1d/ExpandDims/dim�
mnist/fc_9/conv1d/ExpandDims
ExpandDimsmnist/fc_8/Squeeze:output:0)mnist/fc_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
mnist/fc_9/conv1d/ExpandDims�
-mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp�
"mnist/fc_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mnist/fc_9/conv1d/ExpandDims_1/dim�
mnist/fc_9/conv1d/ExpandDims_1
ExpandDims5mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp:value:0+mnist/fc_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2 
mnist/fc_9/conv1d/ExpandDims_1�
mnist/fc_9/conv1dConv2D%mnist/fc_9/conv1d/ExpandDims:output:0'mnist/fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
mnist/fc_9/conv1d�
mnist/fc_9/conv1d/SqueezeSqueezemnist/fc_9/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
mnist/fc_9/conv1d/Squeeze�
!mnist/fc_9/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!mnist/fc_9/BiasAdd/ReadVariableOp�
mnist/fc_9/BiasAddBiasAdd"mnist/fc_9/conv1d/Squeeze:output:0)mnist/fc_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
mnist/fc_9/BiasAdd~
mnist/fc_9/ReluRelumnist/fc_9/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
mnist/fc_9/Relu�
!mnist/fc_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!mnist/fc_10/conv1d/ExpandDims/dim�
mnist/fc_10/conv1d/ExpandDims
ExpandDimsmnist/fc_9/Relu:activations:0*mnist/fc_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
mnist/fc_10/conv1d/ExpandDims�
.mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp7mnist_fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype020
.mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp�
#mnist/fc_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#mnist/fc_10/conv1d/ExpandDims_1/dim�
mnist/fc_10/conv1d/ExpandDims_1
ExpandDims6mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp:value:0,mnist/fc_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2!
mnist/fc_10/conv1d/ExpandDims_1�
mnist/fc_10/conv1dConv2D&mnist/fc_10/conv1d/ExpandDims:output:0(mnist/fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
mnist/fc_10/conv1d�
mnist/fc_10/conv1d/SqueezeSqueezemnist/fc_10/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
mnist/fc_10/conv1d/Squeeze�
"mnist/fc_10/BiasAdd/ReadVariableOpReadVariableOp+mnist_fc_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"mnist/fc_10/BiasAdd/ReadVariableOp�
mnist/fc_10/BiasAddBiasAdd#mnist/fc_10/conv1d/Squeeze:output:0*mnist/fc_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
mnist/fc_10/BiasAdd�
mnist/fc_10/ReluRelumnist/fc_10/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
mnist/fc_10/Relu�
mnist/fc_11/IdentityIdentitymnist/fc_10/Relu:activations:0*
T0*,
_output_shapes
:����������2
mnist/fc_11/Identityz
mnist/fc_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
mnist/fc_12/ExpandDims/dim�
mnist/fc_12/ExpandDims
ExpandDimsmnist/fc_11/Identity:output:0#mnist/fc_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
mnist/fc_12/ExpandDims�
mnist/fc_12/MaxPoolMaxPoolmnist/fc_12/ExpandDims:output:0*/
_output_shapes
:���������^*
ksize
*
paddingVALID*
strides
2
mnist/fc_12/MaxPool�
mnist/fc_12/SqueezeSqueezemnist/fc_12/MaxPool:output:0*
T0*+
_output_shapes
:���������^*
squeeze_dims
2
mnist/fc_12/Squeezeu
mnist/fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
mnist/fc13/Const�
mnist/fc13/ReshapeReshapemnist/fc_12/Squeeze:output:0mnist/fc13/Const:output:0*
T0*(
_output_shapes
:����������2
mnist/fc13/Reshape�
"mnist/output/MatMul/ReadVariableOpReadVariableOp+mnist_output_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02$
"mnist/output/MatMul/ReadVariableOp�
mnist/output/MatMulMatMulmnist/fc13/Reshape:output:0*mnist/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
mnist/output/MatMul�
#mnist/output/BiasAdd/ReadVariableOpReadVariableOp,mnist_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#mnist/output/BiasAdd/ReadVariableOp�
mnist/output/BiasAddBiasAddmnist/output/MatMul:product:0+mnist/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
mnist/output/BiasAdd�
mnist/output/SoftmaxSoftmaxmnist/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
mnist/output/Softmax�
IdentityIdentitymnist/output/Softmax:softmax:0"^mnist/fc_1/BiasAdd/ReadVariableOp.^mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp#^mnist/fc_10/BiasAdd/ReadVariableOp/^mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_2/BiasAdd/ReadVariableOp.^mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_5/BiasAdd/ReadVariableOp.^mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_6/BiasAdd/ReadVariableOp.^mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_9/BiasAdd/ReadVariableOp.^mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp$^mnist/output/BiasAdd/ReadVariableOp#^mnist/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2F
!mnist/fc_1/BiasAdd/ReadVariableOp!mnist/fc_1/BiasAdd/ReadVariableOp2^
-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp2H
"mnist/fc_10/BiasAdd/ReadVariableOp"mnist/fc_10/BiasAdd/ReadVariableOp2`
.mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp.mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_2/BiasAdd/ReadVariableOp!mnist/fc_2/BiasAdd/ReadVariableOp2^
-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_5/BiasAdd/ReadVariableOp!mnist/fc_5/BiasAdd/ReadVariableOp2^
-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_6/BiasAdd/ReadVariableOp!mnist/fc_6/BiasAdd/ReadVariableOp2^
-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_9/BiasAdd/ReadVariableOp!mnist/fc_9/BiasAdd/ReadVariableOp2^
-mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp2J
#mnist/output/BiasAdd/ReadVariableOp#mnist/output/BiasAdd/ReadVariableOp2H
"mnist/output/MatMul/ReadVariableOp"mnist/output/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
{
&__inference_fc_2_layer_call_fn_2775498

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27754912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
`
B__inference_fc_11_layer_call_and_return_conditional_losses_2776031

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_fc_4_layer_call_and_return_conditional_losses_2776090

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
_
A__inference_fc_7_layer_call_and_return_conditional_losses_2775503

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
B
&__inference_fc_4_layer_call_fn_2776095

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_27760902
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
�
'__inference_mnist_layer_call_fn_2776488	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27764692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
�
A__inference_fc_5_layer_call_and_return_conditional_losses_2775810

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_5/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_5/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
@
%__inference_signature_wrapper_2777539	
input
identity�
PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *#
fR
__inference_pruned_27775322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
�
A__inference_fc_9_layer_call_and_return_conditional_losses_2775292

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_9/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_9/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
B
&__inference_fc_8_layer_call_fn_2776512

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27763272
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
�
'__inference_mnist_layer_call_fn_2776507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27764692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
B
&__inference_fc_3_layer_call_fn_2775307

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27753022
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�y
�
B__inference_mnist_layer_call_and_return_conditional_losses_2776469

inputs
fc_1_2074658
fc_1_2074660
fc_2_2074663
fc_2_2074665
fc_5_2074670
fc_5_2074672
fc_6_2074675
fc_6_2074677
fc_9_2074682
fc_9_2074684
fc_10_2074687
fc_10_2074689
output_2074695
output_2074697
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_10/StatefulPartitionedCall�.fc_10/kernel/Regularizer/Square/ReadVariableOp�fc_11/StatefulPartitionedCall�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_3/StatefulPartitionedCall�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_7/StatefulPartitionedCall�fc_9/StatefulPartitionedCall�-fc_9/kernel/Regularizer/Square/ReadVariableOp�output/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_27762412
flatten/PartitionedCall�
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_3/ExpandDims/dim�
tf.expand_dims_3/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_3/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_3/ExpandDims:output:0fc_1_2074658fc_1_2074660*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27754622
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2074663fc_2_2074665*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27754912
fc_2/StatefulPartitionedCall�
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27760772
fc_3/StatefulPartitionedCall�
fc_4/PartitionedCallPartitionedCall%fc_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_27760902
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2074670fc_5_2074672*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27753512
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2074675fc_6_2074677*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27763122
fc_6/StatefulPartitionedCall�
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0^fc_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27754352
fc_7/StatefulPartitionedCall�
fc_8/PartitionedCallPartitionedCall%fc_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27763272
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2074682fc_9_2074684*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27760092
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2074687fc_10_2074689*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27755302
fc_10/StatefulPartitionedCall�
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0^fc_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27761072
fc_11/StatefulPartitionedCall�
fc_12/PartitionedCallPartitionedCall&fc_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27757442
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27757832
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2074695output_2074697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_27757702 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2074658*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2074663*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2074670*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2074675*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2074682*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2074687*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_10/StatefulPartitionedCall/^fc_10/kernel/Regularizer/Square/ReadVariableOp^fc_11/StatefulPartitionedCall^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_3/StatefulPartitionedCall^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_7/StatefulPartitionedCall^fc_9/StatefulPartitionedCall.^fc_9/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp2>
fc_11/StatefulPartitionedCallfc_11/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
{
&__inference_fc_9_layer_call_fn_2776016

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27760092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
{
&__inference_fc_6_layer_call_fn_2776319

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27763122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_2775270:
6fc_1_kernel_regularizer_square_readvariableop_resource
identity��-fc_1/kernel/Regularizer/Square/ReadVariableOp�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
IdentityIdentityfc_1/kernel/Regularizer/mul:z:0.^fc_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp
��
�
B__inference_mnist_layer_call_and_return_conditional_losses_2775987

inputs4
0fc_1_conv1d_expanddims_1_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource4
0fc_2_conv1d_expanddims_1_readvariableop_resource(
$fc_2_biasadd_readvariableop_resource4
0fc_5_conv1d_expanddims_1_readvariableop_resource(
$fc_5_biasadd_readvariableop_resource4
0fc_6_conv1d_expanddims_1_readvariableop_resource(
$fc_6_biasadd_readvariableop_resource4
0fc_9_conv1d_expanddims_1_readvariableop_resource(
$fc_9_biasadd_readvariableop_resource5
1fc_10_conv1d_expanddims_1_readvariableop_resource)
%fc_10_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��fc_1/BiasAdd/ReadVariableOp�'fc_1/conv1d/ExpandDims_1/ReadVariableOp�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_10/BiasAdd/ReadVariableOp�(fc_10/conv1d/ExpandDims_1/ReadVariableOp�.fc_10/kernel/Regularizer/Square/ReadVariableOp�fc_2/BiasAdd/ReadVariableOp�'fc_2/conv1d/ExpandDims_1/ReadVariableOp�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/BiasAdd/ReadVariableOp�'fc_5/conv1d/ExpandDims_1/ReadVariableOp�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/BiasAdd/ReadVariableOp�'fc_6/conv1d/ExpandDims_1/ReadVariableOp�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_9/BiasAdd/ReadVariableOp�'fc_9/conv1d/ExpandDims_1/ReadVariableOp�-fc_9/kernel/Regularizer/Square/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
flatten/Const�
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_3/ExpandDims/dim�
tf.expand_dims_3/ExpandDims
ExpandDimsflatten/Reshape:output:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_3/ExpandDims�
fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_1/conv1d/ExpandDims/dim�
fc_1/conv1d/ExpandDims
ExpandDims$tf.expand_dims_3/ExpandDims:output:0#fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_1/conv1d/ExpandDims�
'fc_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02)
'fc_1/conv1d/ExpandDims_1/ReadVariableOp~
fc_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_1/conv1d/ExpandDims_1/dim�
fc_1/conv1d/ExpandDims_1
ExpandDims/fc_1/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
fc_1/conv1d/ExpandDims_1�
fc_1/conv1dConv2Dfc_1/conv1d/ExpandDims:output:0!fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_1/conv1d�
fc_1/conv1d/SqueezeSqueezefc_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_1/conv1d/Squeeze�
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_1/BiasAdd/ReadVariableOp�
fc_1/BiasAddBiasAddfc_1/conv1d/Squeeze:output:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_1/BiasAddl
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
	fc_1/Relu�
fc_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_2/conv1d/ExpandDims/dim�
fc_2/conv1d/ExpandDims
ExpandDimsfc_1/Relu:activations:0#fc_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
fc_2/conv1d/ExpandDims�
'fc_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02)
'fc_2/conv1d/ExpandDims_1/ReadVariableOp~
fc_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_2/conv1d/ExpandDims_1/dim�
fc_2/conv1d/ExpandDims_1
ExpandDims/fc_2/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
fc_2/conv1d/ExpandDims_1�
fc_2/conv1dConv2Dfc_2/conv1d/ExpandDims:output:0!fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_2/conv1d�
fc_2/conv1d/SqueezeSqueezefc_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_2/conv1d/Squeeze�
fc_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_2/BiasAdd/ReadVariableOp�
fc_2/BiasAddBiasAddfc_2/conv1d/Squeeze:output:0#fc_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_2/BiasAddl
	fc_2/ReluRelufc_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
	fc_2/Reluz
fc_3/IdentityIdentityfc_2/Relu:activations:0*
T0*,
_output_shapes
:����������@2
fc_3/Identityl
fc_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
fc_4/ExpandDims/dim�
fc_4/ExpandDims
ExpandDimsfc_3/Identity:output:0fc_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
fc_4/ExpandDims�
fc_4/MaxPoolMaxPoolfc_4/ExpandDims:output:0*0
_output_shapes
:����������@*
ksize
*
paddingVALID*
strides
2
fc_4/MaxPool�
fc_4/SqueezeSqueezefc_4/MaxPool:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2
fc_4/Squeeze�
fc_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_5/conv1d/ExpandDims/dim�
fc_5/conv1d/ExpandDims
ExpandDimsfc_4/Squeeze:output:0#fc_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
fc_5/conv1d/ExpandDims�
'fc_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02)
'fc_5/conv1d/ExpandDims_1/ReadVariableOp~
fc_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_5/conv1d/ExpandDims_1/dim�
fc_5/conv1d/ExpandDims_1
ExpandDims/fc_5/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
fc_5/conv1d/ExpandDims_1�
fc_5/conv1dConv2Dfc_5/conv1d/ExpandDims:output:0!fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_5/conv1d�
fc_5/conv1d/SqueezeSqueezefc_5/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_5/conv1d/Squeeze�
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_5/BiasAdd/ReadVariableOp�
fc_5/BiasAddBiasAddfc_5/conv1d/Squeeze:output:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_5/BiasAddl
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
	fc_5/Relu�
fc_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_6/conv1d/ExpandDims/dim�
fc_6/conv1d/ExpandDims
ExpandDimsfc_5/Relu:activations:0#fc_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
fc_6/conv1d/ExpandDims�
'fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02)
'fc_6/conv1d/ExpandDims_1/ReadVariableOp~
fc_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_6/conv1d/ExpandDims_1/dim�
fc_6/conv1d/ExpandDims_1
ExpandDims/fc_6/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
fc_6/conv1d/ExpandDims_1�
fc_6/conv1dConv2Dfc_6/conv1d/ExpandDims:output:0!fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_6/conv1d�
fc_6/conv1d/SqueezeSqueezefc_6/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_6/conv1d/Squeeze�
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_6/BiasAdd/ReadVariableOp�
fc_6/BiasAddBiasAddfc_6/conv1d/Squeeze:output:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_6/BiasAddl
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
	fc_6/Reluz
fc_7/IdentityIdentityfc_6/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
fc_7/Identityl
fc_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
fc_8/ExpandDims/dim�
fc_8/ExpandDims
ExpandDimsfc_7/Identity:output:0fc_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
fc_8/ExpandDims�
fc_8/MaxPoolMaxPoolfc_8/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingVALID*
strides
2
fc_8/MaxPool�
fc_8/SqueezeSqueezefc_8/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2
fc_8/Squeeze�
fc_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_9/conv1d/ExpandDims/dim�
fc_9/conv1d/ExpandDims
ExpandDimsfc_8/Squeeze:output:0#fc_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
fc_9/conv1d/ExpandDims�
'fc_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02)
'fc_9/conv1d/ExpandDims_1/ReadVariableOp~
fc_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_9/conv1d/ExpandDims_1/dim�
fc_9/conv1d/ExpandDims_1
ExpandDims/fc_9/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
fc_9/conv1d/ExpandDims_1�
fc_9/conv1dConv2Dfc_9/conv1d/ExpandDims:output:0!fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_9/conv1d�
fc_9/conv1d/SqueezeSqueezefc_9/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_9/conv1d/Squeeze�
fc_9/BiasAdd/ReadVariableOpReadVariableOp$fc_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_9/BiasAdd/ReadVariableOp�
fc_9/BiasAddBiasAddfc_9/conv1d/Squeeze:output:0#fc_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_9/BiasAddl
	fc_9/ReluRelufc_9/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
	fc_9/Relu�
fc_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_10/conv1d/ExpandDims/dim�
fc_10/conv1d/ExpandDims
ExpandDimsfc_9/Relu:activations:0$fc_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_10/conv1d/ExpandDims�
(fc_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02*
(fc_10/conv1d/ExpandDims_1/ReadVariableOp�
fc_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_10/conv1d/ExpandDims_1/dim�
fc_10/conv1d/ExpandDims_1
ExpandDims0fc_10/conv1d/ExpandDims_1/ReadVariableOp:value:0&fc_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
fc_10/conv1d/ExpandDims_1�
fc_10/conv1dConv2D fc_10/conv1d/ExpandDims:output:0"fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_10/conv1d�
fc_10/conv1d/SqueezeSqueezefc_10/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_10/conv1d/Squeeze�
fc_10/BiasAdd/ReadVariableOpReadVariableOp%fc_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_10/BiasAdd/ReadVariableOp�
fc_10/BiasAddBiasAddfc_10/conv1d/Squeeze:output:0$fc_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_10/BiasAddo

fc_10/ReluRelufc_10/BiasAdd:output:0*
T0*,
_output_shapes
:����������2

fc_10/Relu}
fc_11/IdentityIdentityfc_10/Relu:activations:0*
T0*,
_output_shapes
:����������2
fc_11/Identityn
fc_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
fc_12/ExpandDims/dim�
fc_12/ExpandDims
ExpandDimsfc_11/Identity:output:0fc_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_12/ExpandDims�
fc_12/MaxPoolMaxPoolfc_12/ExpandDims:output:0*/
_output_shapes
:���������^*
ksize
*
paddingVALID*
strides
2
fc_12/MaxPool�
fc_12/SqueezeSqueezefc_12/MaxPool:output:0*
T0*+
_output_shapes
:���������^*
squeeze_dims
2
fc_12/Squeezei

fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2

fc13/Const�
fc13/ReshapeReshapefc_12/Squeeze:output:0fc13/Const:output:0*
T0*(
_output_shapes
:����������2
fc13/Reshape�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMulfc13/Reshape:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
output/Softmax�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentityoutput/Softmax:softmax:0^fc_1/BiasAdd/ReadVariableOp(^fc_1/conv1d/ExpandDims_1/ReadVariableOp.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_10/BiasAdd/ReadVariableOp)^fc_10/conv1d/ExpandDims_1/ReadVariableOp/^fc_10/kernel/Regularizer/Square/ReadVariableOp^fc_2/BiasAdd/ReadVariableOp(^fc_2/conv1d/ExpandDims_1/ReadVariableOp.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/BiasAdd/ReadVariableOp(^fc_5/conv1d/ExpandDims_1/ReadVariableOp.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/BiasAdd/ReadVariableOp(^fc_6/conv1d/ExpandDims_1/ReadVariableOp.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_9/BiasAdd/ReadVariableOp(^fc_9/conv1d/ExpandDims_1/ReadVariableOp.^fc_9/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2R
'fc_1/conv1d/ExpandDims_1/ReadVariableOp'fc_1/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2<
fc_10/BiasAdd/ReadVariableOpfc_10/BiasAdd/ReadVariableOp2T
(fc_10/conv1d/ExpandDims_1/ReadVariableOp(fc_10/conv1d/ExpandDims_1/ReadVariableOp2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp2:
fc_2/BiasAdd/ReadVariableOpfc_2/BiasAdd/ReadVariableOp2R
'fc_2/conv1d/ExpandDims_1/ReadVariableOp'fc_2/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2:
fc_5/BiasAdd/ReadVariableOpfc_5/BiasAdd/ReadVariableOp2R
'fc_5/conv1d/ExpandDims_1/ReadVariableOp'fc_5/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2:
fc_6/BiasAdd/ReadVariableOpfc_6/BiasAdd/ReadVariableOp2R
'fc_6/conv1d/ExpandDims_1/ReadVariableOp'fc_6/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2:
fc_9/BiasAdd/ReadVariableOpfc_9/BiasAdd/ReadVariableOp2R
'fc_9/conv1d/ExpandDims_1/ReadVariableOp'fc_9/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_fc_2_layer_call_and_return_conditional_losses_2775329

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_2/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_2/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
A__inference_fc_6_layer_call_and_return_conditional_losses_2775730

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_6/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_6/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2776713

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
A__inference_fc_7_layer_call_and_return_conditional_losses_2776258

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
��
�
B__inference_mnist_layer_call_and_return_conditional_losses_2775698

inputs4
0fc_1_conv1d_expanddims_1_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource4
0fc_2_conv1d_expanddims_1_readvariableop_resource(
$fc_2_biasadd_readvariableop_resource4
0fc_5_conv1d_expanddims_1_readvariableop_resource(
$fc_5_biasadd_readvariableop_resource4
0fc_6_conv1d_expanddims_1_readvariableop_resource(
$fc_6_biasadd_readvariableop_resource4
0fc_9_conv1d_expanddims_1_readvariableop_resource(
$fc_9_biasadd_readvariableop_resource5
1fc_10_conv1d_expanddims_1_readvariableop_resource)
%fc_10_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��fc_1/BiasAdd/ReadVariableOp�'fc_1/conv1d/ExpandDims_1/ReadVariableOp�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_10/BiasAdd/ReadVariableOp�(fc_10/conv1d/ExpandDims_1/ReadVariableOp�.fc_10/kernel/Regularizer/Square/ReadVariableOp�fc_2/BiasAdd/ReadVariableOp�'fc_2/conv1d/ExpandDims_1/ReadVariableOp�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/BiasAdd/ReadVariableOp�'fc_5/conv1d/ExpandDims_1/ReadVariableOp�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/BiasAdd/ReadVariableOp�'fc_6/conv1d/ExpandDims_1/ReadVariableOp�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_9/BiasAdd/ReadVariableOp�'fc_9/conv1d/ExpandDims_1/ReadVariableOp�-fc_9/kernel/Regularizer/Square/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
flatten/Const�
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_3/ExpandDims/dim�
tf.expand_dims_3/ExpandDims
ExpandDimsflatten/Reshape:output:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_3/ExpandDims�
fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_1/conv1d/ExpandDims/dim�
fc_1/conv1d/ExpandDims
ExpandDims$tf.expand_dims_3/ExpandDims:output:0#fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_1/conv1d/ExpandDims�
'fc_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02)
'fc_1/conv1d/ExpandDims_1/ReadVariableOp~
fc_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_1/conv1d/ExpandDims_1/dim�
fc_1/conv1d/ExpandDims_1
ExpandDims/fc_1/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
fc_1/conv1d/ExpandDims_1�
fc_1/conv1dConv2Dfc_1/conv1d/ExpandDims:output:0!fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_1/conv1d�
fc_1/conv1d/SqueezeSqueezefc_1/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_1/conv1d/Squeeze�
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_1/BiasAdd/ReadVariableOp�
fc_1/BiasAddBiasAddfc_1/conv1d/Squeeze:output:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_1/BiasAddl
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
	fc_1/Relu�
fc_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_2/conv1d/ExpandDims/dim�
fc_2/conv1d/ExpandDims
ExpandDimsfc_1/Relu:activations:0#fc_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
fc_2/conv1d/ExpandDims�
'fc_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02)
'fc_2/conv1d/ExpandDims_1/ReadVariableOp~
fc_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_2/conv1d/ExpandDims_1/dim�
fc_2/conv1d/ExpandDims_1
ExpandDims/fc_2/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
fc_2/conv1d/ExpandDims_1�
fc_2/conv1dConv2Dfc_2/conv1d/ExpandDims:output:0!fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_2/conv1d�
fc_2/conv1d/SqueezeSqueezefc_2/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_2/conv1d/Squeeze�
fc_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_2/BiasAdd/ReadVariableOp�
fc_2/BiasAddBiasAddfc_2/conv1d/Squeeze:output:0#fc_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_2/BiasAddl
	fc_2/ReluRelufc_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
	fc_2/Relum
fc_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
fc_3/dropout/Const�
fc_3/dropout/MulMulfc_2/Relu:activations:0fc_3/dropout/Const:output:0*
T0*,
_output_shapes
:����������@2
fc_3/dropout/Mulo
fc_3/dropout/ShapeShapefc_2/Relu:activations:0*
T0*
_output_shapes
:2
fc_3/dropout/Shape�
)fc_3/dropout/random_uniform/RandomUniformRandomUniformfc_3/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype02+
)fc_3/dropout/random_uniform/RandomUniform
fc_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
fc_3/dropout/GreaterEqual/y�
fc_3/dropout/GreaterEqualGreaterEqual2fc_3/dropout/random_uniform/RandomUniform:output:0$fc_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
fc_3/dropout/GreaterEqual�
fc_3/dropout/CastCastfc_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
fc_3/dropout/Cast�
fc_3/dropout/Mul_1Mulfc_3/dropout/Mul:z:0fc_3/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
fc_3/dropout/Mul_1l
fc_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
fc_4/ExpandDims/dim�
fc_4/ExpandDims
ExpandDimsfc_3/dropout/Mul_1:z:0fc_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
fc_4/ExpandDims�
fc_4/MaxPoolMaxPoolfc_4/ExpandDims:output:0*0
_output_shapes
:����������@*
ksize
*
paddingVALID*
strides
2
fc_4/MaxPool�
fc_4/SqueezeSqueezefc_4/MaxPool:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2
fc_4/Squeeze�
fc_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_5/conv1d/ExpandDims/dim�
fc_5/conv1d/ExpandDims
ExpandDimsfc_4/Squeeze:output:0#fc_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
fc_5/conv1d/ExpandDims�
'fc_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02)
'fc_5/conv1d/ExpandDims_1/ReadVariableOp~
fc_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_5/conv1d/ExpandDims_1/dim�
fc_5/conv1d/ExpandDims_1
ExpandDims/fc_5/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
fc_5/conv1d/ExpandDims_1�
fc_5/conv1dConv2Dfc_5/conv1d/ExpandDims:output:0!fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_5/conv1d�
fc_5/conv1d/SqueezeSqueezefc_5/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_5/conv1d/Squeeze�
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_5/BiasAdd/ReadVariableOp�
fc_5/BiasAddBiasAddfc_5/conv1d/Squeeze:output:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_5/BiasAddl
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
	fc_5/Relu�
fc_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_6/conv1d/ExpandDims/dim�
fc_6/conv1d/ExpandDims
ExpandDimsfc_5/Relu:activations:0#fc_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
fc_6/conv1d/ExpandDims�
'fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02)
'fc_6/conv1d/ExpandDims_1/ReadVariableOp~
fc_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_6/conv1d/ExpandDims_1/dim�
fc_6/conv1d/ExpandDims_1
ExpandDims/fc_6/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
fc_6/conv1d/ExpandDims_1�
fc_6/conv1dConv2Dfc_6/conv1d/ExpandDims:output:0!fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_6/conv1d�
fc_6/conv1d/SqueezeSqueezefc_6/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_6/conv1d/Squeeze�
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_6/BiasAdd/ReadVariableOp�
fc_6/BiasAddBiasAddfc_6/conv1d/Squeeze:output:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_6/BiasAddl
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
	fc_6/Relum
fc_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
fc_7/dropout/Const�
fc_7/dropout/MulMulfc_6/Relu:activations:0fc_7/dropout/Const:output:0*
T0*,
_output_shapes
:���������� 2
fc_7/dropout/Mulo
fc_7/dropout/ShapeShapefc_6/Relu:activations:0*
T0*
_output_shapes
:2
fc_7/dropout/Shape�
)fc_7/dropout/random_uniform/RandomUniformRandomUniformfc_7/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
dtype02+
)fc_7/dropout/random_uniform/RandomUniform
fc_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
fc_7/dropout/GreaterEqual/y�
fc_7/dropout/GreaterEqualGreaterEqual2fc_7/dropout/random_uniform/RandomUniform:output:0$fc_7/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������� 2
fc_7/dropout/GreaterEqual�
fc_7/dropout/CastCastfc_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
fc_7/dropout/Cast�
fc_7/dropout/Mul_1Mulfc_7/dropout/Mul:z:0fc_7/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
fc_7/dropout/Mul_1l
fc_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
fc_8/ExpandDims/dim�
fc_8/ExpandDims
ExpandDimsfc_7/dropout/Mul_1:z:0fc_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
fc_8/ExpandDims�
fc_8/MaxPoolMaxPoolfc_8/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingVALID*
strides
2
fc_8/MaxPool�
fc_8/SqueezeSqueezefc_8/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2
fc_8/Squeeze�
fc_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_9/conv1d/ExpandDims/dim�
fc_9/conv1d/ExpandDims
ExpandDimsfc_8/Squeeze:output:0#fc_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
fc_9/conv1d/ExpandDims�
'fc_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02)
'fc_9/conv1d/ExpandDims_1/ReadVariableOp~
fc_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_9/conv1d/ExpandDims_1/dim�
fc_9/conv1d/ExpandDims_1
ExpandDims/fc_9/conv1d/ExpandDims_1/ReadVariableOp:value:0%fc_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
fc_9/conv1d/ExpandDims_1�
fc_9/conv1dConv2Dfc_9/conv1d/ExpandDims:output:0!fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_9/conv1d�
fc_9/conv1d/SqueezeSqueezefc_9/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_9/conv1d/Squeeze�
fc_9/BiasAdd/ReadVariableOpReadVariableOp$fc_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_9/BiasAdd/ReadVariableOp�
fc_9/BiasAddBiasAddfc_9/conv1d/Squeeze:output:0#fc_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_9/BiasAddl
	fc_9/ReluRelufc_9/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
	fc_9/Relu�
fc_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_10/conv1d/ExpandDims/dim�
fc_10/conv1d/ExpandDims
ExpandDimsfc_9/Relu:activations:0$fc_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_10/conv1d/ExpandDims�
(fc_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02*
(fc_10/conv1d/ExpandDims_1/ReadVariableOp�
fc_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
fc_10/conv1d/ExpandDims_1/dim�
fc_10/conv1d/ExpandDims_1
ExpandDims0fc_10/conv1d/ExpandDims_1/ReadVariableOp:value:0&fc_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
fc_10/conv1d/ExpandDims_1�
fc_10/conv1dConv2D fc_10/conv1d/ExpandDims:output:0"fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_10/conv1d�
fc_10/conv1d/SqueezeSqueezefc_10/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_10/conv1d/Squeeze�
fc_10/BiasAdd/ReadVariableOpReadVariableOp%fc_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_10/BiasAdd/ReadVariableOp�
fc_10/BiasAddBiasAddfc_10/conv1d/Squeeze:output:0$fc_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_10/BiasAddo

fc_10/ReluRelufc_10/BiasAdd:output:0*
T0*,
_output_shapes
:����������2

fc_10/Reluo
fc_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
fc_11/dropout/Const�
fc_11/dropout/MulMulfc_10/Relu:activations:0fc_11/dropout/Const:output:0*
T0*,
_output_shapes
:����������2
fc_11/dropout/Mulr
fc_11/dropout/ShapeShapefc_10/Relu:activations:0*
T0*
_output_shapes
:2
fc_11/dropout/Shape�
*fc_11/dropout/random_uniform/RandomUniformRandomUniformfc_11/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02,
*fc_11/dropout/random_uniform/RandomUniform�
fc_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
fc_11/dropout/GreaterEqual/y�
fc_11/dropout/GreaterEqualGreaterEqual3fc_11/dropout/random_uniform/RandomUniform:output:0%fc_11/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
fc_11/dropout/GreaterEqual�
fc_11/dropout/CastCastfc_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
fc_11/dropout/Cast�
fc_11/dropout/Mul_1Mulfc_11/dropout/Mul:z:0fc_11/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
fc_11/dropout/Mul_1n
fc_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
fc_12/ExpandDims/dim�
fc_12/ExpandDims
ExpandDimsfc_11/dropout/Mul_1:z:0fc_12/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_12/ExpandDims�
fc_12/MaxPoolMaxPoolfc_12/ExpandDims:output:0*/
_output_shapes
:���������^*
ksize
*
paddingVALID*
strides
2
fc_12/MaxPool�
fc_12/SqueezeSqueezefc_12/MaxPool:output:0*
T0*+
_output_shapes
:���������^*
squeeze_dims
2
fc_12/Squeezei

fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2

fc13/Const�
fc13/ReshapeReshapefc_12/Squeeze:output:0fc13/Const:output:0*
T0*(
_output_shapes
:����������2
fc13/Reshape�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMulfc13/Reshape:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
output/Softmax�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentityoutput/Softmax:softmax:0^fc_1/BiasAdd/ReadVariableOp(^fc_1/conv1d/ExpandDims_1/ReadVariableOp.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_10/BiasAdd/ReadVariableOp)^fc_10/conv1d/ExpandDims_1/ReadVariableOp/^fc_10/kernel/Regularizer/Square/ReadVariableOp^fc_2/BiasAdd/ReadVariableOp(^fc_2/conv1d/ExpandDims_1/ReadVariableOp.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/BiasAdd/ReadVariableOp(^fc_5/conv1d/ExpandDims_1/ReadVariableOp.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/BiasAdd/ReadVariableOp(^fc_6/conv1d/ExpandDims_1/ReadVariableOp.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_9/BiasAdd/ReadVariableOp(^fc_9/conv1d/ExpandDims_1/ReadVariableOp.^fc_9/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2R
'fc_1/conv1d/ExpandDims_1/ReadVariableOp'fc_1/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2<
fc_10/BiasAdd/ReadVariableOpfc_10/BiasAdd/ReadVariableOp2T
(fc_10/conv1d/ExpandDims_1/ReadVariableOp(fc_10/conv1d/ExpandDims_1/ReadVariableOp2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp2:
fc_2/BiasAdd/ReadVariableOpfc_2/BiasAdd/ReadVariableOp2R
'fc_2/conv1d/ExpandDims_1/ReadVariableOp'fc_2/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2:
fc_5/BiasAdd/ReadVariableOpfc_5/BiasAdd/ReadVariableOp2R
'fc_5/conv1d/ExpandDims_1/ReadVariableOp'fc_5/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2:
fc_6/BiasAdd/ReadVariableOpfc_6/BiasAdd/ReadVariableOp2R
'fc_6/conv1d/ExpandDims_1/ReadVariableOp'fc_6/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2:
fc_9/BiasAdd/ReadVariableOpfc_9/BiasAdd/ReadVariableOp2R
'fc_9/conv1d/ExpandDims_1/ReadVariableOp'fc_9/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2775770

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_fc_2_layer_call_and_return_conditional_losses_2775491

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_2/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_2/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�<
�
#__inference__traced_restore_2777656
file_prefix 
assignvariableop_fc_1_kernel 
assignvariableop_1_fc_1_bias"
assignvariableop_2_fc_2_kernel 
assignvariableop_3_fc_2_bias"
assignvariableop_4_fc_5_kernel 
assignvariableop_5_fc_5_bias"
assignvariableop_6_fc_6_kernel 
assignvariableop_7_fc_6_bias"
assignvariableop_8_fc_9_kernel 
assignvariableop_9_fc_9_bias$
 assignvariableop_10_fc_10_kernel"
assignvariableop_11_fc_10_bias%
!assignvariableop_12_output_kernel#
assignvariableop_13_output_bias
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_fc_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_fc_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_fc_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_fc_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_fc_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_fc_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_fc_9_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_fc_9_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp assignvariableop_10_fc_10_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_fc_10_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14�
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�t
�
B__inference_mnist_layer_call_and_return_conditional_losses_2776583

inputs
fc_1_2074776
fc_1_2074778
fc_2_2074781
fc_2_2074783
fc_5_2074788
fc_5_2074790
fc_6_2074793
fc_6_2074795
fc_9_2074800
fc_9_2074802
fc_10_2074805
fc_10_2074807
output_2074813
output_2074815
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_10/StatefulPartitionedCall�.fc_10/kernel/Regularizer/Square/ReadVariableOp�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_9/StatefulPartitionedCall�-fc_9/kernel/Regularizer/Square/ReadVariableOp�output/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_27762412
flatten/PartitionedCall�
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_3/ExpandDims/dim�
tf.expand_dims_3/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_3/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_3/ExpandDims:output:0fc_1_2074776fc_1_2074778*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27754622
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2074781fc_2_2074783*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27754912
fc_2/StatefulPartitionedCall�
fc_3/PartitionedCallPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27753022
fc_3/PartitionedCall�
fc_4/PartitionedCallPartitionedCallfc_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_27760902
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2074788fc_5_2074790*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27753512
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2074793fc_6_2074795*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27763122
fc_6/StatefulPartitionedCall�
fc_7/PartitionedCallPartitionedCall%fc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27755032
fc_7/PartitionedCall�
fc_8/PartitionedCallPartitionedCallfc_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27763272
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2074800fc_9_2074802*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27760092
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2074805fc_10_2074807*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27755302
fc_10/StatefulPartitionedCall�
fc_11/PartitionedCallPartitionedCall&fc_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27757032
fc_11/PartitionedCall�
fc_12/PartitionedCallPartitionedCallfc_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27757442
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27757832
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2074813output_2074815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_27757702 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2074776*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2074781*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2074788*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2074793*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2074800*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2074805*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_10/StatefulPartitionedCall/^fc_10/kernel/Regularizer/Square/ReadVariableOp^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_9/StatefulPartitionedCall.^fc_9/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
'__inference_fc_10_layer_call_fn_2775537

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27755302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�'
�
 __inference__traced_save_2777604
file_prefix*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop*
&savev2_fc_2_kernel_read_readvariableop(
$savev2_fc_2_bias_read_readvariableop*
&savev2_fc_5_kernel_read_readvariableop(
$savev2_fc_5_bias_read_readvariableop*
&savev2_fc_6_kernel_read_readvariableop(
$savev2_fc_6_bias_read_readvariableop*
&savev2_fc_9_kernel_read_readvariableop(
$savev2_fc_9_bias_read_readvariableop+
'savev2_fc_10_kernel_read_readvariableop)
%savev2_fc_10_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop&savev2_fc_5_kernel_read_readvariableop$savev2_fc_5_bias_read_readvariableop&savev2_fc_6_kernel_read_readvariableop$savev2_fc_6_bias_read_readvariableop&savev2_fc_9_kernel_read_readvariableop$savev2_fc_9_bias_read_readvariableop'savev2_fc_10_kernel_read_readvariableop%savev2_fc_10_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@@:@:@ : :  : : ::::	�
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :(	$
"
_output_shapes
: : 


_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�
: 

_output_shapes
:
:

_output_shapes
: 
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2775847

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
B
&__inference_fc_7_layer_call_fn_2775508

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27755032
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
B
&__inference_fc13_layer_call_fn_2775788

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27757832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������^:S O
+
_output_shapes
:���������^
 
_user_specified_nameinputs
�
�
B__inference_fc_10_layer_call_and_return_conditional_losses_2776053

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�.fc_10/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp/^fc_10/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
A__inference_fc_3_layer_call_and_return_conditional_losses_2776077

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
�
'__inference_mnist_layer_call_fn_2776621

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27765832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_fc13_layer_call_and_return_conditional_losses_2775736

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������^:S O
+
_output_shapes
:���������^
 
_user_specified_nameinputs
�
_
&__inference_fc_3_layer_call_fn_2776082

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27760772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
�
'__inference_mnist_layer_call_fn_2776602	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27765832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
C
'__inference_fc_12_layer_call_fn_2775749

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27757442
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
A__inference_fc_5_layer_call_and_return_conditional_losses_2775351

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_5/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_5/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
`
A__inference_fc_3_layer_call_and_return_conditional_losses_2776725

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
B__inference_fc_10_layer_call_and_return_conditional_losses_2775530

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�.fc_10/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp/^fc_10/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�y
�
B__inference_mnist_layer_call_and_return_conditional_losses_2776692	
input
fc_1_2074198
fc_1_2074200
fc_2_2074236
fc_2_2074238
fc_5_2074305
fc_5_2074307
fc_6_2074343
fc_6_2074345
fc_9_2074412
fc_9_2074414
fc_10_2074450
fc_10_2074452
output_2074522
output_2074524
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_10/StatefulPartitionedCall�.fc_10/kernel/Regularizer/Square/ReadVariableOp�fc_11/StatefulPartitionedCall�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_3/StatefulPartitionedCall�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_7/StatefulPartitionedCall�fc_9/StatefulPartitionedCall�-fc_9/kernel/Regularizer/Square/ReadVariableOp�output/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_27762412
flatten/PartitionedCall�
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_3/ExpandDims/dim�
tf.expand_dims_3/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_3/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_3/ExpandDims:output:0fc_1_2074198fc_1_2074200*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27754622
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2074236fc_2_2074238*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27754912
fc_2/StatefulPartitionedCall�
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27760772
fc_3/StatefulPartitionedCall�
fc_4/PartitionedCallPartitionedCall%fc_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_27760902
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2074305fc_5_2074307*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27753512
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2074343fc_6_2074345*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27763122
fc_6/StatefulPartitionedCall�
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0^fc_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27754352
fc_7/StatefulPartitionedCall�
fc_8/PartitionedCallPartitionedCall%fc_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27763272
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2074412fc_9_2074414*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27760092
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2074450fc_10_2074452*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27755302
fc_10/StatefulPartitionedCall�
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0^fc_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27761072
fc_11/StatefulPartitionedCall�
fc_12/PartitionedCallPartitionedCall&fc_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27757442
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27757832
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2074522output_2074524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_27757702 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2074198*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2074236*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2074305*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2074343*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2074412*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2074450*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_10/StatefulPartitionedCall/^fc_10/kernel/Regularizer/Square/ReadVariableOp^fc_11/StatefulPartitionedCall^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_3/StatefulPartitionedCall^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_7/StatefulPartitionedCall^fc_9/StatefulPartitionedCall.^fc_9/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp2>
fc_11/StatefulPartitionedCallfc_11/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
]
A__inference_fc13_layer_call_and_return_conditional_losses_2775783

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������^:S O
+
_output_shapes
:���������^
 
_user_specified_nameinputs
�t
�
B__inference_mnist_layer_call_and_return_conditional_losses_2776398	
input
fc_1_2074570
fc_1_2074572
fc_2_2074575
fc_2_2074577
fc_5_2074582
fc_5_2074584
fc_6_2074587
fc_6_2074589
fc_9_2074594
fc_9_2074596
fc_10_2074599
fc_10_2074601
output_2074607
output_2074609
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_10/StatefulPartitionedCall�.fc_10/kernel/Regularizer/Square/ReadVariableOp�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_9/StatefulPartitionedCall�-fc_9/kernel/Regularizer/Square/ReadVariableOp�output/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_27762412
flatten/PartitionedCall�
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_3/ExpandDims/dim�
tf.expand_dims_3/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_3/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_3/ExpandDims:output:0fc_1_2074570fc_1_2074572*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27754622
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2074575fc_2_2074577*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27754912
fc_2/StatefulPartitionedCall�
fc_3/PartitionedCallPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27753022
fc_3/PartitionedCall�
fc_4/PartitionedCallPartitionedCallfc_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_27760902
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2074582fc_5_2074584*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27753512
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2074587fc_6_2074589*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27763122
fc_6/StatefulPartitionedCall�
fc_7/PartitionedCallPartitionedCall%fc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27755032
fc_7/PartitionedCall�
fc_8/PartitionedCallPartitionedCallfc_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27763272
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2074594fc_9_2074596*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27760092
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2074599fc_10_2074601*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27755302
fc_10/StatefulPartitionedCall�
fc_11/PartitionedCallPartitionedCall&fc_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27757032
fc_11/PartitionedCall�
fc_12/PartitionedCallPartitionedCallfc_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27757442
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27757832
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2074607output_2074609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_27757702 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2074570*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2074575*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2074582*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2074587*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2074594*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2074599*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_10/StatefulPartitionedCall/^fc_10/kernel/Regularizer/Square/ReadVariableOp^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_9/StatefulPartitionedCall.^fc_9/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
_
A__inference_fc_3_layer_call_and_return_conditional_losses_2775297

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
A__inference_fc_1_layer_call_and_return_conditional_losses_2775462

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_1/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_2776702:
6fc_2_kernel_regularizer_square_readvariableop_resource
identity��-fc_2/kernel/Regularizer/Square/ReadVariableOp�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_2_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
fc_2/kernel/Regularizer/Square�
fc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_2/kernel/Regularizer/Const�
fc_2/kernel/Regularizer/SumSum"fc_2/kernel/Regularizer/Square:y:0&fc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/Sum�
fc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_2/kernel/Regularizer/mul/x�
fc_2/kernel/Regularizer/mulMul&fc_2/kernel/Regularizer/mul/x:output:0$fc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_2/kernel/Regularizer/mul�
IdentityIdentityfc_2/kernel/Regularizer/mul:z:0.^fc_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2776241

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_2776268:
6fc_5_kernel_regularizer_square_readvariableop_resource
identity��-fc_5/kernel/Regularizer/Square/ReadVariableOp�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_5_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
fc_5/kernel/Regularizer/Square�
fc_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_5/kernel/Regularizer/Const�
fc_5/kernel/Regularizer/SumSum"fc_5/kernel/Regularizer/Square:y:0&fc_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/Sum�
fc_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_5/kernel/Regularizer/mul/x�
fc_5/kernel/Regularizer/mulMul&fc_5/kernel/Regularizer/mul/x:output:0$fc_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_5/kernel/Regularizer/mul�
IdentityIdentityfc_5/kernel/Regularizer/mul:z:0.^fc_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_3_2775368:
6fc_6_kernel_regularizer_square_readvariableop_resource
identity��-fc_6/kernel/Regularizer/Square/ReadVariableOp�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_6_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
IdentityIdentityfc_6/kernel/Regularizer/mul:z:0.^fc_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp
�
_
&__inference_fc_7_layer_call_fn_2775440

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27754352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
a
B__inference_fc_11_layer_call_and_return_conditional_losses_2776065

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_fc_9_layer_call_and_return_conditional_losses_2776009

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_9/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_9/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
`
B__inference_fc_11_layer_call_and_return_conditional_losses_2775703

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_output_layer_call_fn_2775777

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_27757702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_2776026;
7fc_10_kernel_regularizer_square_readvariableop_resource
identity��.fc_10/kernel/Regularizer/Square/ReadVariableOp�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7fc_10_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:*
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2!
fc_10/kernel/Regularizer/Square�
fc_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2 
fc_10/kernel/Regularizer/Const�
fc_10/kernel/Regularizer/SumSum#fc_10/kernel/Regularizer/Square:y:0'fc_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/Sum�
fc_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
fc_10/kernel/Regularizer/mul/x�
fc_10/kernel/Regularizer/mulMul'fc_10/kernel/Regularizer/mul/x:output:0%fc_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_10/kernel/Regularizer/mul�
IdentityIdentity fc_10/kernel/Regularizer/mul:z:0/^fc_10/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp
�
_
A__inference_fc_3_layer_call_and_return_conditional_losses_2775302

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_4_2775759:
6fc_9_kernel_regularizer_square_readvariableop_resource
identity��-fc_9/kernel/Regularizer/Square/ReadVariableOp�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_9_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
fc_9/kernel/Regularizer/Square�
fc_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_9/kernel/Regularizer/Const�
fc_9/kernel/Regularizer/SumSum"fc_9/kernel/Regularizer/Square:y:0&fc_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/Sum�
fc_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_9/kernel/Regularizer/mul/x�
fc_9/kernel/Regularizer/mulMul&fc_9/kernel/Regularizer/mul/x:output:0$fc_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_9/kernel/Regularizer/mul�
IdentityIdentityfc_9/kernel/Regularizer/mul:z:0.^fc_9/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp
�
�
A__inference_fc_1_layer_call_and_return_conditional_losses_2776290

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@2 
fc_1/kernel/Regularizer/Square�
fc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_1/kernel/Regularizer/Const�
fc_1/kernel/Regularizer/SumSum"fc_1/kernel/Regularizer/Square:y:0&fc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/Sum�
fc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_1/kernel/Regularizer/mul/x�
fc_1/kernel/Regularizer/mulMul&fc_1/kernel/Regularizer/mul/x:output:0$fc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_1/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_fc_6_layer_call_and_return_conditional_losses_2776312

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�-fc_6/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
fc_6/kernel/Regularizer/Square�
fc_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
fc_6/kernel/Regularizer/Const�
fc_6/kernel/Regularizer/SumSum"fc_6/kernel/Regularizer/Square:y:0&fc_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/Sum�
fc_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
fc_6/kernel/Regularizer/mul/x�
fc_6/kernel/Regularizer/mulMul&fc_6/kernel/Regularizer/mul/x:output:0$fc_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
fc_6/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp.^fc_6/kernel/Regularizer/Square/ReadVariableOp*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
a
B__inference_fc_11_layer_call_and_return_conditional_losses_2776107

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_fc_12_layer_call_and_return_conditional_losses_2775744

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
_
A__inference_fc_7_layer_call_and_return_conditional_losses_2775373

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
]
A__inference_fc_8_layer_call_and_return_conditional_losses_2776327

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
{
&__inference_fc_1_layer_call_fn_2775469

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27754622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
A__inference_fc_7_layer_call_and_return_conditional_losses_2775435

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
{
&__inference_fc_5_layer_call_fn_2775358

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27753512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_2776246

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_27762412
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_fc_11_layer_call_fn_2775708

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27757032
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
'__inference_fc_11_layer_call_fn_2776112

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27761072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input6
serving_default_input:0���������2
output(
PartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
regularization_losses
trainable_variables
	variables
	keras_api

signatures
#_self_saveable_object_factories
trt_engine_resources
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"
_generic_user_object
C
#_self_saveable_object_factories"
_generic_user_object
�
regularization_losses
trainable_variables
	variables
	keras_api
#_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
R
	keras_api
# _self_saveable_object_factories"
_generic_user_object
�

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
#'_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
#._self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
/regularization_losses
0trainable_variables
1	variables
2	keras_api
#3_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
4regularization_losses
5trainable_variables
6	variables
7	keras_api
#8_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

9kernel
:bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
#?_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
#F_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
#K_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
#P_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
#W_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

Xkernel
Ybias
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
#^_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
_regularization_losses
`trainable_variables
a	variables
b	keras_api
#c_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
dregularization_losses
etrainable_variables
f	variables
g	keras_api
#h_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
#m_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

nkernel
obias
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
#t_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
�
!0
"1
(2
)3
94
:5
@6
A7
Q8
R9
X10
Y11
n12
o13"
trackable_list_wrapper
�
!0
"1
(2
)3
94
:5
@6
A7
Q8
R9
X10
Y11
n12
o13"
trackable_list_wrapper
�
unon_trainable_variables
regularization_losses
vlayer_metrics
wmetrics
xlayer_regularization_losses
trainable_variables

ylayers
	variables
#z_self_saveable_object_factories
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{non_trainable_variables
regularization_losses
|layer_metrics
}metrics
~layer_regularization_losses
trainable_variables

layers
	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
D
$�_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
!:@2fc_1/kernel
:@2	fc_1/bias
(
�0"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
�
�non_trainable_variables
#regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
$trainable_variables
�layers
%	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!:@@2fc_2/kernel
:@2	fc_2/bias
(
�0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
�non_trainable_variables
*regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
+trainable_variables
�layers
,	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
/regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
0trainable_variables
�layers
1	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
4regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
5trainable_variables
�layers
6	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!:@ 2fc_5/kernel
: 2	fc_5/bias
(
�0"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
�
�non_trainable_variables
;regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
<trainable_variables
�layers
=	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!:  2fc_6/kernel
: 2	fc_6/bias
(
�0"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
�
�non_trainable_variables
Bregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Ctrainable_variables
�layers
D	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Gregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Htrainable_variables
�layers
I	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Lregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Mtrainable_variables
�layers
N	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!: 2fc_9/kernel
:2	fc_9/bias
(
�0"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
�non_trainable_variables
Sregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Ttrainable_variables
�layers
U	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 2fc_10/kernel
:2
fc_10/bias
(
�0"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
�
�non_trainable_variables
Zregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
[trainable_variables
�layers
\	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
_regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
`trainable_variables
�layers
a	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
dregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
etrainable_variables
�layers
f	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
iregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
jtrainable_variables
�layers
k	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 :	�
2output/kernel
:
2output/bias
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
�
�non_trainable_variables
pregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
qtrainable_variables
�layers
r	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
'__inference_mnist_layer_call_fn_2776602
'__inference_mnist_layer_call_fn_2776507
'__inference_mnist_layer_call_fn_2776488
'__inference_mnist_layer_call_fn_2776621�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_2776216�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *,�)
'�$
input���������
�2�
B__inference_mnist_layer_call_and_return_conditional_losses_2776692
B__inference_mnist_layer_call_and_return_conditional_losses_2775698
B__inference_mnist_layer_call_and_return_conditional_losses_2775987
B__inference_mnist_layer_call_and_return_conditional_losses_2776398�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_flatten_layer_call_fn_2776246�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_layer_call_and_return_conditional_losses_2775847�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_fc_1_layer_call_fn_2775469�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_fc_1_layer_call_and_return_conditional_losses_2776290�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_fc_2_layer_call_fn_2775498�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_fc_2_layer_call_and_return_conditional_losses_2775329�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_fc_3_layer_call_fn_2775307
&__inference_fc_3_layer_call_fn_2776082�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_fc_3_layer_call_and_return_conditional_losses_2775297
A__inference_fc_3_layer_call_and_return_conditional_losses_2776725�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_fc_4_layer_call_fn_2776095�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
A__inference_fc_4_layer_call_and_return_conditional_losses_2776090�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
&__inference_fc_5_layer_call_fn_2775358�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_fc_5_layer_call_and_return_conditional_losses_2775810�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_fc_6_layer_call_fn_2776319�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_fc_6_layer_call_and_return_conditional_losses_2775730�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_fc_7_layer_call_fn_2775440
&__inference_fc_7_layer_call_fn_2775508�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_fc_7_layer_call_and_return_conditional_losses_2775373
A__inference_fc_7_layer_call_and_return_conditional_losses_2776258�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_fc_8_layer_call_fn_2776512�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
A__inference_fc_8_layer_call_and_return_conditional_losses_2776327�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
&__inference_fc_9_layer_call_fn_2776016�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_fc_9_layer_call_and_return_conditional_losses_2775292�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_fc_10_layer_call_fn_2775537�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_fc_10_layer_call_and_return_conditional_losses_2776053�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_fc_11_layer_call_fn_2776112
'__inference_fc_11_layer_call_fn_2775708�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_fc_11_layer_call_and_return_conditional_losses_2776065
B__inference_fc_11_layer_call_and_return_conditional_losses_2776031�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_fc_12_layer_call_fn_2775749�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
B__inference_fc_12_layer_call_and_return_conditional_losses_2775744�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
&__inference_fc13_layer_call_fn_2775788�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_fc13_layer_call_and_return_conditional_losses_2775736�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_output_layer_call_fn_2775777�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_output_layer_call_and_return_conditional_losses_2776713�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_2775270�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_2776702�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_2776268�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_2775368�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_2775759�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_5_2776026�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
%__inference_signature_wrapper_2777539input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_2776216y!"()9:@AQRXYno6�3
,�)
'�$
input���������
� "/�,
*
output �
output���������
�
A__inference_fc13_layer_call_and_return_conditional_losses_2775736]3�0
)�&
$�!
inputs���������^
� "&�#
�
0����������
� z
&__inference_fc13_layer_call_fn_2775788P3�0
)�&
$�!
inputs���������^
� "������������
B__inference_fc_10_layer_call_and_return_conditional_losses_2776053fXY4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
'__inference_fc_10_layer_call_fn_2775537YXY4�1
*�'
%�"
inputs����������
� "������������
B__inference_fc_11_layer_call_and_return_conditional_losses_2776031f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
B__inference_fc_11_layer_call_and_return_conditional_losses_2776065f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
'__inference_fc_11_layer_call_fn_2775708Y8�5
.�+
%�"
inputs����������
p 
� "������������
'__inference_fc_11_layer_call_fn_2776112Y8�5
.�+
%�"
inputs����������
p
� "������������
B__inference_fc_12_layer_call_and_return_conditional_losses_2775744�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
'__inference_fc_12_layer_call_fn_2775749wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_1_layer_call_and_return_conditional_losses_2776290f!"4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
&__inference_fc_1_layer_call_fn_2775469Y!"4�1
*�'
%�"
inputs����������
� "�����������@�
A__inference_fc_2_layer_call_and_return_conditional_losses_2775329f()4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
&__inference_fc_2_layer_call_fn_2775498Y()4�1
*�'
%�"
inputs����������@
� "�����������@�
A__inference_fc_3_layer_call_and_return_conditional_losses_2775297f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
A__inference_fc_3_layer_call_and_return_conditional_losses_2776725f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
&__inference_fc_3_layer_call_fn_2775307Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
&__inference_fc_3_layer_call_fn_2776082Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
A__inference_fc_4_layer_call_and_return_conditional_losses_2776090�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_fc_4_layer_call_fn_2776095wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_5_layer_call_and_return_conditional_losses_2775810f9:4�1
*�'
%�"
inputs����������@
� "*�'
 �
0���������� 
� �
&__inference_fc_5_layer_call_fn_2775358Y9:4�1
*�'
%�"
inputs����������@
� "����������� �
A__inference_fc_6_layer_call_and_return_conditional_losses_2775730f@A4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0���������� 
� �
&__inference_fc_6_layer_call_fn_2776319Y@A4�1
*�'
%�"
inputs���������� 
� "����������� �
A__inference_fc_7_layer_call_and_return_conditional_losses_2775373f8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
A__inference_fc_7_layer_call_and_return_conditional_losses_2776258f8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
&__inference_fc_7_layer_call_fn_2775440Y8�5
.�+
%�"
inputs���������� 
p
� "����������� �
&__inference_fc_7_layer_call_fn_2775508Y8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
A__inference_fc_8_layer_call_and_return_conditional_losses_2776327�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_fc_8_layer_call_fn_2776512wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_9_layer_call_and_return_conditional_losses_2775292fQR4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0����������
� �
&__inference_fc_9_layer_call_fn_2776016YQR4�1
*�'
%�"
inputs���������� 
� "������������
D__inference_flatten_layer_call_and_return_conditional_losses_2775847a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
)__inference_flatten_layer_call_fn_2776246T7�4
-�*
(�%
inputs���������
� "�����������<
__inference_loss_fn_0_2775270!�

� 
� "� <
__inference_loss_fn_1_2776702(�

� 
� "� <
__inference_loss_fn_2_27762689�

� 
� "� <
__inference_loss_fn_3_2775368@�

� 
� "� <
__inference_loss_fn_4_2775759Q�

� 
� "� <
__inference_loss_fn_5_2776026X�

� 
� "� �
B__inference_mnist_layer_call_and_return_conditional_losses_2775698x!"()9:@AQRXYno?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������

� �
B__inference_mnist_layer_call_and_return_conditional_losses_2775987x!"()9:@AQRXYno?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������

� �
B__inference_mnist_layer_call_and_return_conditional_losses_2776398w!"()9:@AQRXYno>�;
4�1
'�$
input���������
p 

 
� "%�"
�
0���������

� �
B__inference_mnist_layer_call_and_return_conditional_losses_2776692w!"()9:@AQRXYno>�;
4�1
'�$
input���������
p

 
� "%�"
�
0���������

� �
'__inference_mnist_layer_call_fn_2776488j!"()9:@AQRXYno>�;
4�1
'�$
input���������
p

 
� "����������
�
'__inference_mnist_layer_call_fn_2776507k!"()9:@AQRXYno?�<
5�2
(�%
inputs���������
p

 
� "����������
�
'__inference_mnist_layer_call_fn_2776602j!"()9:@AQRXYno>�;
4�1
'�$
input���������
p 

 
� "����������
�
'__inference_mnist_layer_call_fn_2776621k!"()9:@AQRXYno?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
C__inference_output_layer_call_and_return_conditional_losses_2776713]no0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� |
(__inference_output_layer_call_fn_2775777Pno0�-
&�#
!�
inputs����������
� "����������
�
%__inference_signature_wrapper_2777539r?�<
� 
5�2
0
input'�$
input���������"/�,
*
output �
output���������
