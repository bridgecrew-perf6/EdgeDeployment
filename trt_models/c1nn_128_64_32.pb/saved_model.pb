��1
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
 �"serve*2.4.12unknown8��/
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
shape:@@*
shared_namefc_5/kernel
o
fc_5/kernel/Read/ReadVariableOpReadVariableOpfc_5/kernel*"
_output_shapes
:@@*
dtype0
j
	fc_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_5/bias
c
fc_5/bias/Read/ReadVariableOpReadVariableOp	fc_5/bias*
_output_shapes
:@*
dtype0
v
fc_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namefc_6/kernel
o
fc_6/kernel/Read/ReadVariableOpReadVariableOpfc_6/kernel*"
_output_shapes
:@@*
dtype0
j
	fc_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_6/bias
c
fc_6/bias/Read/ReadVariableOpReadVariableOp	fc_6/bias*
_output_shapes
:@*
dtype0
v
fc_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_namefc_9/kernel
o
fc_9/kernel/Read/ReadVariableOpReadVariableOpfc_9/kernel*"
_output_shapes
:@ *
dtype0
j
	fc_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	fc_9/bias
c
fc_9/bias/Read/ReadVariableOpReadVariableOp	fc_9/bias*
_output_shapes
: *
dtype0
x
fc_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namefc_10/kernel
q
 fc_10/kernel/Read/ReadVariableOpReadVariableOpfc_10/kernel*"
_output_shapes
:  *
dtype0
l

fc_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
fc_10/bias
e
fc_10/bias/Read/ReadVariableOpReadVariableOp
fc_10/bias*
_output_shapes
: *
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�
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
%__inference_signature_wrapper_2782477
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
 __inference__traced_save_2782542
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
#__inference__traced_restore_2782594��.
�
{
&__inference_fc_9_layer_call_fn_2781235

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
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27812282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
]
A__inference_fc13_layer_call_and_return_conditional_losses_2780215

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������^ :S O
+
_output_shapes
:���������^ 
 
_user_specified_nameinputs
�
�
A__inference_fc_5_layer_call_and_return_conditional_losses_2781257

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
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:����������@2

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
�
'__inference_mnist_layer_call_fn_2781466

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
B__inference_mnist_layer_call_and_return_conditional_losses_27814282
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
��
�

"__inference__wrapped_model_2781148	
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
%mnist/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%mnist/tf.expand_dims_4/ExpandDims/dim�
!mnist/tf.expand_dims_4/ExpandDims
ExpandDimsmnist/flatten/Reshape:output:0.mnist/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2#
!mnist/tf.expand_dims_4/ExpandDims�
 mnist/fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_1/conv1d/ExpandDims/dim�
mnist/fc_1/conv1d/ExpandDims
ExpandDims*mnist/tf.expand_dims_4/ExpandDims:output:0)mnist/fc_1/conv1d/ExpandDims/dim:output:0*
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
:@@*
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
:@@2 
mnist/fc_5/conv1d/ExpandDims_1�
mnist/fc_5/conv1dConv2D%mnist/fc_5/conv1d/ExpandDims:output:0'mnist/fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
mnist/fc_5/conv1d�
mnist/fc_5/conv1d/SqueezeSqueezemnist/fc_5/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
mnist/fc_5/conv1d/Squeeze�
!mnist/fc_5/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!mnist/fc_5/BiasAdd/ReadVariableOp�
mnist/fc_5/BiasAddBiasAdd"mnist/fc_5/conv1d/Squeeze:output:0)mnist/fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
mnist/fc_5/BiasAdd~
mnist/fc_5/ReluRelumnist/fc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
mnist/fc_6/conv1d/ExpandDims�
-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2 
mnist/fc_6/conv1d/ExpandDims_1�
mnist/fc_6/conv1dConv2D%mnist/fc_6/conv1d/ExpandDims:output:0'mnist/fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
mnist/fc_6/conv1d�
mnist/fc_6/conv1d/SqueezeSqueezemnist/fc_6/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
mnist/fc_6/conv1d/Squeeze�
!mnist/fc_6/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!mnist/fc_6/BiasAdd/ReadVariableOp�
mnist/fc_6/BiasAddBiasAdd"mnist/fc_6/conv1d/Squeeze:output:0)mnist/fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
mnist/fc_6/BiasAdd~
mnist/fc_6/ReluRelumnist/fc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
mnist/fc_6/Relu�
mnist/fc_7/IdentityIdentitymnist/fc_6/Relu:activations:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
mnist/fc_8/ExpandDims�
mnist/fc_8/MaxPoolMaxPoolmnist/fc_8/ExpandDims:output:0*0
_output_shapes
:����������@*
ksize
*
paddingVALID*
strides
2
mnist/fc_8/MaxPool�
mnist/fc_8/SqueezeSqueezemnist/fc_8/MaxPool:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
mnist/fc_9/conv1d/ExpandDims�
-mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
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
:@ 2 
mnist/fc_9/conv1d/ExpandDims_1�
mnist/fc_9/conv1dConv2D%mnist/fc_9/conv1d/ExpandDims:output:0'mnist/fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
mnist/fc_9/conv1d�
mnist/fc_9/conv1d/SqueezeSqueezemnist/fc_9/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
mnist/fc_9/conv1d/Squeeze�
!mnist/fc_9/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!mnist/fc_9/BiasAdd/ReadVariableOp�
mnist/fc_9/BiasAddBiasAdd"mnist/fc_9/conv1d/Squeeze:output:0)mnist/fc_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_9/BiasAdd~
mnist/fc_9/ReluRelumnist/fc_9/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
mnist/fc_10/conv1d/ExpandDims�
.mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp7mnist_fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2!
mnist/fc_10/conv1d/ExpandDims_1�
mnist/fc_10/conv1dConv2D&mnist/fc_10/conv1d/ExpandDims:output:0(mnist/fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
mnist/fc_10/conv1d�
mnist/fc_10/conv1d/SqueezeSqueezemnist/fc_10/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
mnist/fc_10/conv1d/Squeeze�
"mnist/fc_10/BiasAdd/ReadVariableOpReadVariableOp+mnist_fc_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"mnist/fc_10/BiasAdd/ReadVariableOp�
mnist/fc_10/BiasAddBiasAdd#mnist/fc_10/conv1d/Squeeze:output:0*mnist/fc_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_10/BiasAdd�
mnist/fc_10/ReluRelumnist/fc_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_10/Relu�
mnist/fc_11/IdentityIdentitymnist/fc_10/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
mnist/fc_12/ExpandDims�
mnist/fc_12/MaxPoolMaxPoolmnist/fc_12/ExpandDims:output:0*/
_output_shapes
:���������^ *
ksize
*
paddingVALID*
strides
2
mnist/fc_12/MaxPool�
mnist/fc_12/SqueezeSqueezemnist/fc_12/MaxPool:output:0*
T0*+
_output_shapes
:���������^ *
squeeze_dims
2
mnist/fc_12/Squeezeu
mnist/fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
mnist/fc13/Const�
mnist/fc13/ReshapeReshapemnist/fc_12/Squeeze:output:0mnist/fc13/Const:output:0*
T0*(
_output_shapes
:����������2
mnist/fc13/Reshape�
"mnist/output/MatMul/ReadVariableOpReadVariableOp+mnist_output_matmul_readvariableop_resource*
_output_shapes
:	�
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
�
�
A__inference_fc_5_layer_call_and_return_conditional_losses_2781032

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
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:����������@2

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
�
{
&__inference_fc_5_layer_call_fn_2781039

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
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27810322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�y
�
B__inference_mnist_layer_call_and_return_conditional_losses_2781428

inputs
fc_1_2716129
fc_1_2716131
fc_2_2716134
fc_2_2716136
fc_5_2716141
fc_5_2716143
fc_6_2716146
fc_6_2716148
fc_9_2716153
fc_9_2716155
fc_10_2716158
fc_10_2716160
output_2716166
output_2716168
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
D__inference_flatten_layer_call_and_return_conditional_losses_27802042
flatten/PartitionedCall�
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_4/ExpandDims/dim�
tf.expand_dims_4/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_4/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_4/ExpandDims:output:0fc_1_2716129fc_1_2716131*
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
A__inference_fc_1_layer_call_and_return_conditional_losses_27812792
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2716134fc_2_2716136*
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
A__inference_fc_2_layer_call_and_return_conditional_losses_27807122
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
A__inference_fc_3_layer_call_and_return_conditional_losses_27805932
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27808052
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2716141fc_5_2716143*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27810322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2716146fc_6_2716148*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27806202
fc_6/StatefulPartitionedCall�
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0^fc_3/StatefulPartitionedCall*
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
A__inference_fc_7_layer_call_and_return_conditional_losses_27811912
fc_7/StatefulPartitionedCall�
fc_8/PartitionedCallPartitionedCall%fc_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27804752
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2716153fc_9_2716155*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27812282
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2716158fc_10_2716160*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27806832
fc_10/StatefulPartitionedCall�
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0^fc_7/StatefulPartitionedCall*
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
GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27809822
fc_11/StatefulPartitionedCall�
fc_12/PartitionedCallPartitionedCall&fc_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27810052
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27802152
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2716166output_2716168*
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
C__inference_output_layer_call_and_return_conditional_losses_27807802 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2716129*"
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2716134*"
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2716141*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2716146*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2716153*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2716158*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
�	
�
'__inference_mnist_layer_call_fn_2781646

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
B__inference_mnist_layer_call_and_return_conditional_losses_27816082
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
�
_
A__inference_fc_7_layer_call_and_return_conditional_losses_2781044

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
A__inference_fc_9_layer_call_and_return_conditional_losses_2781228

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
:����������@2
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
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
B
&__inference_fc_7_layer_call_fn_2780997

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
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27809922
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
_
&__inference_fc_7_layer_call_fn_2781196

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
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27811912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_2780517:
6fc_5_kernel_regularizer_square_readvariableop_resource
identity��-fc_5/kernel/Regularizer/Square/ReadVariableOp�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_5_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
�
�
B__inference_fc_10_layer_call_and_return_conditional_losses_2780683

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
:���������� 2
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
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
A__inference_fc_2_layer_call_and_return_conditional_losses_2780712

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
�t
�
B__inference_mnist_layer_call_and_return_conditional_losses_2781608

inputs
fc_1_2716247
fc_1_2716249
fc_2_2716252
fc_2_2716254
fc_5_2716259
fc_5_2716261
fc_6_2716264
fc_6_2716266
fc_9_2716271
fc_9_2716273
fc_10_2716276
fc_10_2716278
output_2716284
output_2716286
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
D__inference_flatten_layer_call_and_return_conditional_losses_27802042
flatten/PartitionedCall�
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_4/ExpandDims/dim�
tf.expand_dims_4/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_4/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_4/ExpandDims:output:0fc_1_2716247fc_1_2716249*
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
A__inference_fc_1_layer_call_and_return_conditional_losses_27812792
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2716252fc_2_2716254*
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
A__inference_fc_2_layer_call_and_return_conditional_losses_27807122
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
A__inference_fc_3_layer_call_and_return_conditional_losses_27805222
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27808052
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2716259fc_5_2716261*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27810322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2716264fc_6_2716266*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27806202
fc_6/StatefulPartitionedCall�
fc_7/PartitionedCallPartitionedCall%fc_6/StatefulPartitionedCall:output:0*
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
A__inference_fc_7_layer_call_and_return_conditional_losses_27809922
fc_7/PartitionedCall�
fc_8/PartitionedCallPartitionedCallfc_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27804752
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2716271fc_9_2716273*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27812282
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2716276fc_10_2716278*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27806832
fc_10/StatefulPartitionedCall�
fc_11/PartitionedCallPartitionedCall&fc_10/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27812012
fc_11/PartitionedCall�
fc_12/PartitionedCallPartitionedCallfc_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27810052
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27802152
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2716284output_2716286*
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
C__inference_output_layer_call_and_return_conditional_losses_27807802 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2716247*"
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2716252*"
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2716259*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2716264*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2716271*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2716276*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2780780

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
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
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
A__inference_fc_7_layer_call_and_return_conditional_losses_2780992

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
{
&__inference_fc_6_layer_call_fn_2780627

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
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27806202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
_
A__inference_fc_3_layer_call_and_return_conditional_losses_2780225

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
�
^
B__inference_fc_12_layer_call_and_return_conditional_losses_2781005

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
`
B__inference_fc_11_layer_call_and_return_conditional_losses_2780507

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�	
�
'__inference_mnist_layer_call_fn_2781627	
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
B__inference_mnist_layer_call_and_return_conditional_losses_27816082
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
�
}
(__inference_output_layer_call_fn_2780787

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
C__inference_output_layer_call_and_return_conditional_losses_27807802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
{
&__inference_fc_1_layer_call_fn_2781357

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
A__inference_fc_1_layer_call_and_return_conditional_losses_27812792
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
�
B
&__inference_fc_4_layer_call_fn_2780810

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
A__inference_fc_4_layer_call_and_return_conditional_losses_27808052
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
�
__inference_loss_fn_4_2780820:
6fc_9_kernel_regularizer_square_readvariableop_resource
identity��-fc_9/kernel/Regularizer/Square/ReadVariableOp�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_9_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
�<
�
#__inference__traced_restore_2782594
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
�
�
A__inference_fc_6_layer_call_and_return_conditional_losses_2780502

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
:����������@2
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
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�y
�
B__inference_mnist_layer_call_and_return_conditional_losses_2781537	
input
fc_1_2715669
fc_1_2715671
fc_2_2715707
fc_2_2715709
fc_5_2715776
fc_5_2715778
fc_6_2715814
fc_6_2715816
fc_9_2715883
fc_9_2715885
fc_10_2715921
fc_10_2715923
output_2715993
output_2715995
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
D__inference_flatten_layer_call_and_return_conditional_losses_27802042
flatten/PartitionedCall�
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_4/ExpandDims/dim�
tf.expand_dims_4/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_4/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_4/ExpandDims:output:0fc_1_2715669fc_1_2715671*
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
A__inference_fc_1_layer_call_and_return_conditional_losses_27812792
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2715707fc_2_2715709*
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
A__inference_fc_2_layer_call_and_return_conditional_losses_27807122
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
A__inference_fc_3_layer_call_and_return_conditional_losses_27805932
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27808052
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2715776fc_5_2715778*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27810322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2715814fc_6_2715816*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27806202
fc_6/StatefulPartitionedCall�
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0^fc_3/StatefulPartitionedCall*
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
A__inference_fc_7_layer_call_and_return_conditional_losses_27811912
fc_7/StatefulPartitionedCall�
fc_8/PartitionedCallPartitionedCall%fc_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27804752
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2715883fc_9_2715885*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27812282
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2715921fc_10_2715923*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27806832
fc_10/StatefulPartitionedCall�
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0^fc_7/StatefulPartitionedCall*
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
GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27809822
fc_11/StatefulPartitionedCall�
fc_12/PartitionedCallPartitionedCall&fc_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27810052
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27802152
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2715993output_2715995*
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
C__inference_output_layer_call_and_return_conditional_losses_27807802 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2715669*"
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2715707*"
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2715776*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2715814*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2715883*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2715921*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
@
%__inference_signature_wrapper_2782477	
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
__inference_pruned_27824702
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
A__inference_fc_1_layer_call_and_return_conditional_losses_2780455

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
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2780204

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
`
A__inference_fc_7_layer_call_and_return_conditional_losses_2781191

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
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
��
�
B__inference_mnist_layer_call_and_return_conditional_losses_2780386

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
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_4/ExpandDims/dim�
tf.expand_dims_4/ExpandDims
ExpandDimsflatten/Reshape:output:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_4/ExpandDims�
fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_1/conv1d/ExpandDims/dim�
fc_1/conv1d/ExpandDims
ExpandDims$tf.expand_dims_4/ExpandDims:output:0#fc_1/conv1d/ExpandDims/dim:output:0*
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
:@@*
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
:@@2
fc_5/conv1d/ExpandDims_1�
fc_5/conv1dConv2Dfc_5/conv1d/ExpandDims:output:0!fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_5/conv1d�
fc_5/conv1d/SqueezeSqueezefc_5/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_5/conv1d/Squeeze�
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_5/BiasAdd/ReadVariableOp�
fc_5/BiasAddBiasAddfc_5/conv1d/Squeeze:output:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_5/BiasAddl
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
fc_6/conv1d/ExpandDims�
'fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2
fc_6/conv1d/ExpandDims_1�
fc_6/conv1dConv2Dfc_6/conv1d/ExpandDims:output:0!fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_6/conv1d�
fc_6/conv1d/SqueezeSqueezefc_6/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_6/conv1d/Squeeze�
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_6/BiasAdd/ReadVariableOp�
fc_6/BiasAddBiasAddfc_6/conv1d/Squeeze:output:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_6/BiasAddl
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
fc_7/dropout/Mulo
fc_7/dropout/ShapeShapefc_6/Relu:activations:0*
T0*
_output_shapes
:2
fc_7/dropout/Shape�
)fc_7/dropout/random_uniform/RandomUniformRandomUniformfc_7/dropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
fc_7/dropout/GreaterEqual�
fc_7/dropout/CastCastfc_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
fc_7/dropout/Cast�
fc_7/dropout/Mul_1Mulfc_7/dropout/Mul:z:0fc_7/dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
fc_8/ExpandDims�
fc_8/MaxPoolMaxPoolfc_8/ExpandDims:output:0*0
_output_shapes
:����������@*
ksize
*
paddingVALID*
strides
2
fc_8/MaxPool�
fc_8/SqueezeSqueezefc_8/MaxPool:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
fc_9/conv1d/ExpandDims�
'fc_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
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
:@ 2
fc_9/conv1d/ExpandDims_1�
fc_9/conv1dConv2Dfc_9/conv1d/ExpandDims:output:0!fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_9/conv1d�
fc_9/conv1d/SqueezeSqueezefc_9/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_9/conv1d/Squeeze�
fc_9/BiasAdd/ReadVariableOpReadVariableOp$fc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_9/BiasAdd/ReadVariableOp�
fc_9/BiasAddBiasAddfc_9/conv1d/Squeeze:output:0#fc_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_9/BiasAddl
	fc_9/ReluRelufc_9/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_10/conv1d/ExpandDims�
(fc_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2
fc_10/conv1d/ExpandDims_1�
fc_10/conv1dConv2D fc_10/conv1d/ExpandDims:output:0"fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_10/conv1d�
fc_10/conv1d/SqueezeSqueezefc_10/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_10/conv1d/Squeeze�
fc_10/BiasAdd/ReadVariableOpReadVariableOp%fc_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_10/BiasAdd/ReadVariableOp�
fc_10/BiasAddBiasAddfc_10/conv1d/Squeeze:output:0$fc_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_10/BiasAddo

fc_10/ReluRelufc_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2

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
:���������� 2
fc_11/dropout/Mulr
fc_11/dropout/ShapeShapefc_10/Relu:activations:0*
T0*
_output_shapes
:2
fc_11/dropout/Shape�
*fc_11/dropout/random_uniform/RandomUniformRandomUniformfc_11/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
fc_11/dropout/GreaterEqual�
fc_11/dropout/CastCastfc_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
fc_11/dropout/Cast�
fc_11/dropout/Mul_1Mulfc_11/dropout/Mul:z:0fc_11/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_12/ExpandDims�
fc_12/MaxPoolMaxPoolfc_12/ExpandDims:output:0*/
_output_shapes
:���������^ *
ksize
*
paddingVALID*
strides
2
fc_12/MaxPool�
fc_12/SqueezeSqueezefc_12/MaxPool:output:0*
T0*+
_output_shapes
:���������^ *
squeeze_dims
2
fc_12/Squeezei

fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2

fc13/Const�
fc13/ReshapeReshapefc_12/Squeeze:output:0fc13/Const:output:0*
T0*(
_output_shapes
:����������2
fc13/Reshape�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�
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
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2781663

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
�
]
A__inference_fc_4_layer_call_and_return_conditional_losses_2780805

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
�
�
A__inference_fc_2_layer_call_and_return_conditional_losses_2780549

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
�
`
A__inference_fc_3_layer_call_and_return_conditional_losses_2780593

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
�
__inference_loss_fn_5_2780970;
7fc_10_kernel_regularizer_square_readvariableop_resource
identity��.fc_10/kernel/Regularizer/Square/ReadVariableOp�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7fc_10_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
�
�
A__inference_fc_6_layer_call_and_return_conditional_losses_2780620

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
:����������@2
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
:����������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������@2
Relu�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:����������@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
Ǟ 
5
__inference_pruned_2782470	
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
=StatefulPartitionedCall/mnist/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2?
=StatefulPartitionedCall/mnist/tf.expand_dims_4/ExpandDims/dim�
9StatefulPartitionedCall/mnist/tf.expand_dims_4/ExpandDims
ExpandDims6StatefulPartitionedCall/mnist/flatten/Reshape:output:0FStatefulPartitionedCall/mnist/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2;
9StatefulPartitionedCall/mnist/tf.expand_dims_4/ExpandDims�
8StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims
ExpandDimsBStatefulPartitionedCall/mnist/tf.expand_dims_4/ExpandDims:output:0AStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dim:output:0*
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
value�B�@*����Q��==��=�ֲ=.���(�����;�f�=/ ��xL������Z�=
I�=��=,Ω=�:��_����<o�����>�)>��y�@��=�-w���=T�q=O@���[_�� ��H�j<rU����ٽ���I��=�Y����==A�<I�g;����K�+�=;߼�F�GI�=�7��q�<9>`�<����=<�=P��<	\�G;,~�����=��(�>�D�=ryc=<>�<�U�9$��r �=�o�=᫆��H�;���=�.">-
�=g�Y=��>�=�<K���#=�]2>Ry�=;�,=
c�=�Ł=��=ˊ<��=�_==_�6�Ӯ4���E=N��=*�=�P�=e�>�˞=��<��P�3�3�_}�����=ё�A�(=R>�=t�=���=��=��b�<F =�S=كʼ�*=��o��(�=.�=��=b�W����=�r�<��f��/t=s܈=L�u=�tt�}�*=z��8��\=��=�>Wߎ;��<�j��+d���f�&�ݽ/����h�=�9�=Y�'���������]�V=�9�Yi�=A�ͽ�t���t����4=������=�r�=��=ON��}�=��;�k=\�ν�R�=���=��=2�3�g`���9�����=_c>�������<�ʿ��] <O��j�>����?�Q��)�=k{�Öp�!Ǐ�C�X���<y���E�=t"����=�&�;���90m=�"5��F>����m(�������>�=^��=��>�i�����28
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
value�B�@*�si=�������:|�;e��;K<(<u�˺��\�i9�<Q���L�ǻ:��:�?�:��[��xꑻ����+Q;F�=;X��:o$ǻ�
d���:�`j���:�N';�׷���;�Ҫ�GT��gt���,O��==耻;�~;��C�	-:.[<�9<    �����D<�Pһw�-9    �~<⥁;P�ѻ��b<��;��8^�o��L�<�0��'<�"��џ�%��G��:,���U�R;JS9;�k��ʩ��2;
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
value��B��@@*���0O��L:�j���L�/t�&�彦6�0�T<�����07���`�H�<@�i���7��qH�mK�=�g��=�A��>���=�Wɼ�.=�B0>�m=��g���
>M��=�U⽌F��Z�^�0ώ�P<��t���T=�|��t�c=RC��r�������B�?ʸ�y'N�J��:�I[>���lrA���.>IMټ#]Ӽj��<\{�Q��<1��= =���꽽�N���X���;<ߕ�s�2��T۽h]�l�!���6�^�J=��� `s<��7��6W�-��%���
9���<�d�=f�;�[�a�b V<�Ҷ=2:�<s��==v�V�/=���=����>���0=˽jKQ<Gg������3�"=WܹTB���<�<.��={�H�c�(:B6e<j�,���<p��< hk=��L=}�׼������I7<�ӼQ�T=�K`�_|G= ��g�<��&=/�ɽ�-=��=��%<��Y=�6\;�Wh����1>���<�����`D��a�;���ʺx;�>r~�<+4�=�Z�<�E =��=1I�=�Z=�%����ͻ��=VRA>��y=n_�<V��=Pߞ=�$>��=|�">�мLT^��N�=���=E���H��>:>��=��=Z��< D����=�[�=o��S�0��g��ҹ�=�{>(��<��=��=Sx<+P��7(�r��=�������<ы�5>��I<5���K�|��=_�=�Ee>�8�=�us�~̎����>)����ĽQ��==׵�=*r�=�q5=q�=� >,��<��J����<Rp;$A3=�=<81����=�M�=F��=\��<V�8����<�m����}=u8�=��Y<���\=�ڐ<�>{=�ڽ!�>gD�<�)������%��=��=�<�c�=XL�<�!�=1��=��>ޥ�=�-<�[�=���j!��^=?�=¿���p<��m=��=�Z�=&�0������=<r=!B>���=�ȧ9�C{�-��=�Ƣ�A�=1�>o��=�<�=^I=�٬�G�=��<E��=d/�<�����<F�T=�>�=��X��3=����)>^��;�D�=��=�=c�5�=D��=��;v&�<u#��@�_<ݟ��_�:}��=�=nĝ=<<�y<:~�;������=5F����H=�=�F�=�%�=�<�0=b��ɨZ���`<�{~=�"<06�=�伹^<U��<k'�a '�d4���㿼5�='EQ=��@=�+߽(1);Z(�=���=�#�=�	?>r����<��M=�[���Fl>�S�<�.=V���6��t��<�!�����XLJ�d��>c��<L���6�;���f=�_���<����/=Y��;�c$=�[���2�<R�>���=dXe�<��z����
�4�<^�(�$�����?�m�w����=�����L'>�*Լ�CX=�T�M��w��<jn^>�=��{>�;��M��=����1T�6�=�l%>�m >J�c=L}>���cG=&s>�3>,�$>:~�+��>�Fa9oq=Ɂ~�8��=�(=>��<���=8��;�a=Y�C�k���o1\<$n�fr4>��=��g��.)=���=QQB�h����}=��#<����8��<���:�;�r���)ې��َ=?x�N@�=c)��&�<�F�<�#-�	�=��<N-�=IE�=S*�<9�����潷EL<#�<�=���C<�v?=6(u��)�<tZ*<������=��d=���=0k�=dc��4s���n>�q<�Iۼ���<�0=��=H���>=�)н~��l�"�<R'�=i��M���cA;O���ҥ�a�e=Ov/���:��S���;���5u>��~�ʱ��rҼ������= Gx�8�սg[;ф��t���h��ؘ8���D�2����)�X���*��Ȧ��y3�c$-��+:n���\}�<q��<:�s���ݷ�=���ۦl��$���0�m�[<G��<<��=ӕ�>*��=���<M\F=&�z�J=�
>TD���L���ܙ���=B�>ti=Y��<sL�sG��c����m;:����㕽������a���O�O����b��|��[g�y/��R
|�ٷ>E���:/����2�x=\�-=����">��>U4��:�
��	=bҀ��N̽*w=1]��vw=[�C=�����m��t��r{����M��[_�0dk=!�>
�ؽ?��{B/>o��<��$P�������<p�缂�}�4�����w����5Y<+�<;`x��X�������:='K.�y"�=⦼�w<��7<�>*<�B��T8�=����,�Y���=4���H=��K<�>����o���<e�;=<lO�W�<bq���=�����φ���T�<��;,�=�dU="0�=�]=��=�z��«<O�<�>;�G۽^��<�Cs=�ډ=o�]<Ϛq���v�^�{�`�L��[7<	���h�=eM�;D
�*�<���=���='�=y�=l��=�+=a��;g =-�<�,�=�p>,��<�<;�I��p&=k�>���<Q�>\�6�Q=s�ҽu[=�='�R9�!������*�>� 亟X{=�;�<���;�Q=��,�;F]7=s傽:ۼ�&ɽ�����=�Ij=�jd=r�<�.ܼS 4=+��=�`μ˓Q��ͽw =���=�(b���
>��	=*�<��������<���>�o=�q�<H�ټ`�y=��=�L���Q=���>��Z>�B�>�
�>�'꼢l�=�|M>?��=���=#@��7�>����/�=+DE��}=�.D�K��=j[�=���<�}�;��%�3���6�Լ��w=?��<�]�/ݢ;(��=ζ=�m��푽�w�=��<=ev<^܎�B	�:+���+���'>(��	�1=�T�=W�o��8U=�]=(]O=p�P��|�=d-��	E�=��-=NU���L@=R��<<<7�%u�=�O)<>޽�^=Zo���C�=ĸN:����g<G���=k'�|:=%�&=7=�I#�5>�p��Z�4U4<k#>P��<�F���=+�j=uY=Z���3#=cĒ<���<�m=g�I<�r=g��=+1�=k>��ʽ��f<�?h<�1>즡;�M��聼,U��T�<�i�A۽٫���=��<W�=1��<����)�<S]<�����j��Ö��$�;�껼
��hX��(��=<�X�&j�JNj<ހ=8�7�P�R�s�<r=�9�Ž��$���;�l>��#<-�>��=��!���=#�>��9)��;^J�����I=�Bc���=~O>u��='B�=�o�=�{N��>��f=�k��#=6˃=|�5=��<��=.#=�^�=�	}��vE=�U�>z����`�S=�p{�����U"g����=�Q=�g<�RK�:�;P����T=�Xüc�����=Y�w=A�=��%=��d<Ԥ�=��ۺ�� �j=b5=���oJD��ս v:=�p�=��i��L��P>ԁ=5�<|=�#��Nk=^�=@o�6�7=T;�=���=G�=ki�����v�ʼ�@�=�}�:�k���=3�]�ʿ�>N
=��c<���=��V>�/{<�k<�{⺅��<�>�>hx��C�.��J������gz���Q��Z��5��<N�*��%=}�{=BeĽD�B��Z=��>H�=��N<����!z�;�3=f�ü�Jk��-=�Sn�3<�<W��= %��DJ2=�	���&����<��><F`�?l>us�=� =��>V������=�LY>>=k�=��Ͻ�W>Y����6K�=�
�=:�>�!a=ր#<
�r=���=u��=�i�=U��<JQ
>�P$>x';��;~^=�͍<�/���$�;�0;=n$*<A�i<���<��=�9�9�|�=�g��V���Y�<,�=P�<��<���=��.�k[>)���#�R��fn=��4=g�=pr >��~=�ڎ�C�"=:Ȧ=��߼u"<������o��b�S�$=<%< �=D%c�0ݻ<�>���E��;��=<��=�15��V= �0><��V�c=��>�)�<�_�ߍ���y˼ww��H�0��<�[�=�DB���=e�=��w*�<!4뽸� �y�=Mi�<�Q�=P�=L⼃j�4X�=�ɤ<hȖ<���=��=z��=ys�U,�=�����=<b޻��K=΋<���"�o{X<�����徼j1�<�z�=_=2� �=~A��A>����<��=�#�����"���]����<�.�<sgҽV����w�5ܔ=��=m�=��s=�O�=�<������=t:��jK>��u<���=*I�=�G��Lv=��O����;�����=m�p=�r�=!1P��x.=)��+��U��~�=�v���ə:�"=�L��,�0� �J�_�ɽ�uy����ױ�=�S�< �Ӽ�U=�9༵��=�J=��=�ZT=�Wy<��=� Y�Wd@3;�0�
��\�����s=a6�<��=3�=����	]�<Y��)t�<,�<�o��&=�r�<x4��AC<�b�=�8=������<@t�9je�<�F�<�9A=BR(>�-f���Ӻ޶r=�P0�G�=���=��=��<C�,=p��=J�=�<��G�=*�I:p�E<z0���k=�P=yOp<PP_�i\b�Km���=D�X=�1�<��=��ݼ��ۼ�H��&.B<�K>��л�=��ȼ:P��R�6�I�V�>~�<��i�O�:��0�zĀ�x:v=�S"=�lz����<gh���r=��<�4��kg�����;�������9��a.���>f&?�4xV=�>��C=5���S��[��<��X:�=������<r��<�	��Oъ=����9��c�l=RBj�')��ǆ=��K�ViH�C<8ּm��8n��1�=�0ȼ��{�stg�$A6=V��a�ѻ���<0�u�[�?=J��P�;=�5)=�ִ<+鱽�m�-!�*_=Z<z
$<��߽m`(<�=	==�`�;���5uL=Iʹ��8�=�5��������;Q�=����
;uY=މ8��Դ<a���L�u=8_��
��<����G�
�:N�/�ٽ*pB�N6:��g��-3�F�=�"��2����<�V��m?n>��ｪJG��V=oɶ�+P
�V�J�|&�ܩ��6��kI�M�ѽ���<I\s�^�����ѼY ��:	��*��q�4���+�
m�<J>$��h;<���<�e��@0�<o��=Nd�������U�	%���P=���F��$U>
u=Ԥ-=U�8=�L�=Nu��}�>�윽�����tk=��9>)���=��C=.8=�K�=W�L;�"λ�*=�n'<��Y=�?��80���[�V��=�= =���/1^=�BE���-��R=-V=B�!�Y��=9��=)'��"=�XݻZZ9=MQ�=��W�c�ؼ�G=���"=�8<=v¨=mY�=5(=�1�<���:�N�<�=�;�
¾
�=߲�;������=3&���=,�����9�MH�G���{t<F~U=@���<��Y�;�v�=�==���=�a�<�)�����<��;!�>�%Q<���}I"=�\�H�=K�$��k*������=���<�‽��s=���e6>�>��1ӽg�=��1���!=E�u=�f{�'ż�>����=�=�����io=�&�<����:�/ݩ=t-��n�>�V<=��)q�<]��=��?=�%�<����eG|�A\�n&���i=)÷�+ƽ6>�x�<��Z����=tlV=��=8��$�K=m��=�?P=����Q��3�\=��=��<8"�=<0�l%;=ۛ�=l�D=H���Q��lx�=?�h�с�=��ayO=���=1��=P�=-�����4�j���D�=s]$�̓�;�V���Dk�=���:=�=�F�=�N%;
�ο��RΑ���{<kxk=��$9�p	���<H�<}��ލ}=�p�=�P<�8�d����o;q�>8�7=V'=����YI=��<mm=��;� �=���<��=E�=`>��/B���>��$=�$�=JW�<TL>���<�d�=a��=pY�=��<z��l;��|<���=T��=�� �EȽ��L=���=��=��=׋2=���<��;>M��r�>g�U;��\�G<A=S�=�+��B�6�}�>�pŽ���=�X%=�7�/#l��x@�5P=<D*��r��]��<�o=�j[=��8=]�=�t<�(���F=�7�=&����ɽ}�=ŗ�=�E���\=�
ٽ���=?1�=��P=>RH<+�'<���=��>�o���}<cv�=)/�=��;^�� ��</W<͚�=��߼_T�ɂa��a=�i�='�q����<$p~=���<"�2=���:B�T<�t�=La�M�<)B�<�h�=u�����<����at<�Q&����=�.,=o9	=����b<./�O҆��r�=7�<��5=������<��/=a�=�t�=�[ڼ �M����<U��8��9��U\��7�=�z=U"8���=f��z��=�B�<��<"aG�0]��Sx�=>�^�'β<�C=;R>)C���:�=��=4<�=���!�;��;r��o�=��<�
{�.R=�2s=H�<�s=}y
=�C�=/�h���m��{�=�F<�T���Φ<c��<އN=�����P#���F�< m�=���T�	��	�=Ϧv<���ȳ�$q�<�R�=:�<���<����@<x�������=e��=��=��=#����=û�<���V:�<�j�=*=���<�-_�ͺ��j66;�6�<��:�s�=+�=�������<�*=���=j�����c>QL�=/�L=ߺȽ7�~=.=�K�;�T.;��I��W�>������	�l�=�9����k�*:&>F(t;;?�9v�5�:=s ˼��^<Ӳ=@>Q=-L�=��=��=��0��F*>�M�=�jJ�t��;)Vq��d>q�=i�=8 �	�\��|�{¤���=Ly�=B(�=BIc>&Ӻ�ד=d�;�K+�{��=H��=,H�;%j>`�{>���=IT�<N�>w�>�B>(�����>��=��u�&1���
4�(�S=]��<�Q����m�����M��k��<�Υ<sAD=�|�<��><��;���DK�.&3�	���������븼z���CX��<s��=�Z�<� �<�?G��n��O`���0�<����
5��"��U��<��^�4>=ܗ�:��H�Ś5��韼�D=VȻ�Q=g;��9<q(@�M��<����<���U}�<.�ӫ<�+
��ͥ<E�B=6b*<�� �l3=q��[��zf=���KK���=�(� �f��~�g-�����=�4���&=5-��h7;� 򺱷;�@<���=��<�r/�+َ��ݫ=��"=3�m=�?S=�ᴼ�F2��_>���=������=K�ʽ��}�<��<`�0��;>=����`Vz�)N�<f�&>���<���;qH�=.æ���V:VhL=i�=�ʻ��2=V'+��j=d�(=��#�gS;=/��;�q�<)�	<��:�N<o	�=E�|;��1�W�<�R�:���c.���ݼ7�
=�2< �T�3q�<�>=&�����<#J='���c=��.=��'����<���;���)@�9}f�<R�_<ُ�����9�o3�<���:�i�;�X���g���&;I�=6�:=�K�@Q#=���<.Ӽ�=���܅�������=��Y�#C=u���jØ��`�<ؐH��jټp�ۼᑋ=���T�9�����k��=b�T=J#o=�<�	��sN<v� =���!]<a5���\o=:Gx=���<�^�;>��=w5�<�=�=�}�=*�a=�	�<��]E�;�=S�+>N������Я�=w�=�����@y=ja�<�F=@:@��c�E0�=$�伞/U=�:X��<Sx[��鿽B�L<j�>��= �<��<�Y=�Z�=�ϛ=���=�]<,=��<г��B=��=��3�aMN=��~<��B=��=[�:�Ƚ� �;'$=[�=!Na>�^�E��<7��>*@�<�>/[�<�0>۸a��0>���=]�A=��S>���=���=?~��μ<>E�B���nʚ�Xi<r�>��=<��'��2�=��j='��O�'>�K�=5P�\��Ն�=^%=�B�=y�S>�`軞�>�|7/>��ȼ�>;���>� �-k����L��|�>�6�=���<�\��b�=���=x\���-9�oP>f�=E&�>�{˼�=�&���T���]6=���l>A��=V%�>Y뼣��=<��C?�=�8�=�u�����>���=�1<��o��yS�8�d=�㘽p�b��E���ؼ�L=)|;</�v:�1>i`����w=`�*��݆���I>	~��_¼�P�=U�����3�߽7�����z�U��:ʇﻺD����o���z�r|���h<o]��3!1�n�e=����J�v=�Q��S����C��=[�����2�"���^��s>Ke�<� 2�y+�<o�J�>{�=7�>F�{=���=4>k̑�3W;�T��>�n�����<[8�<q;A>~t�<:�C<��=��>�"�=�=[5=���<}E�<~m�<i)>���:Ҡ�=�V�=�+>������m4�={;^����<����uИ���^<��7�-�.P�=���}�<qX���<L8�;!�<|�ü�A�Gt=5��7ȟ=�'3=��=��b!ܼ&Δ;`�C�)yh��U=��=����L�9��I�Ƽ�o =P�N�ӣ	���t=���=�i=9�<9ϐ=�=��'�<�q߼���=�\=�2��z�����Z;����:&=[�;�H=k���5��<b����=�i���� �=�l�a�N�^OۼOo���%�=���=���=U63<	���]�
>qG̽���8i���̽>���U�<b#�<N��X�<k�*�7�ǽhO��t.=L�=I�Q��j����C�l=\>��R�=��h�W�`=��+)=����V@�<O���pv�������Z>� �<�u�,?>���S�(=�w>4��=�#L�{�ͼW�9>/���q�꽺@�=o��=C�<؅��䜼�=_�5=l�|�u=�X�<��ߗ=osH= �>/c��=��F<95
��K<`��<��V=�;_�<=R�<�"Y=�Dн>��<�t�$��<��p����=E$���G��:ɒ<�Y��p<�}�=�n���=�MG=k`&=;=�9�ټ��<��%=B�C�=�m����=�:,=�����11=�Ru=�I�;	��=���=[?5��,4�Nw!>��_�PϘ����Xx;�M(�c�=e�<y�;=*;>[��=I�<��H<E'�;�j3;D6�=+��<*�W=kYy=��;o:�:)
�=�佇9׻����D��|�(�;!�<t�[=�쩽6#� =�ײ��;0=�[=qڷ=����ռ�'`��A���J<�-I�|,=��>0�0����=�w��6�D�;��j�=�z���������LJ6��<,YQ�(�9o=;<-g=�6V=7G=�K��oWU����=`c׼�C�����=��<֜�^�=�J�:C �<�z=?��8�9�GH���~_��=��<�7t�{+�=�,�=M��<���=� <S��=��:�qQ*=��E=b�u���=HP���z�;�aQ��>"��=fқ�8Q=�A�����<F=_���Vg=��=�|�=i��;�d=-IF=:��=��)>��G��9�Ok{=�v>r����i<��c�a���=>�� =R�_��,X=�\<��]</֙���"��i�<���<�S�=�P=ڢ�<�B�����:lC��{ZH��(�o�=�������z	�OyA=ǜ�=��"=�/��P��ړJ=����p2=�~=��R=�a��&�hkh��<���_�;�jT=���;;И�[��=��̽��ʽ�c=�=Y�4���뼗ֶ=��˼7�<�ż=�>�=ǻ�T�H=���j{�=�g��w\;+�<h�O�WUD=x�ͼ�.�<(;�@�<%��`P���l�=k��-�x=��k<=�<�&�;�����v�=�{=�6�Oj�<ږ@=*�U�
^��Q���'���=�{�=p��;MZ�=7q��R��<VɄ=����Z�<&C =UCH>��n��<��P=�A�7�@=wԼ�.�}��=;��=��]��*b�d�"=Ǎ���}�|�=�T�홫��f�>q=��4����<*�=s������G�"=d��g�=��'q�jT�<s��;�"Ľ5��]����={����^�=e#+�#�:�ǧ=�J>>� ��o���=B4<��M=2�R�Ʌi=!7�=7mk=�<�=D�t�`�0=pV�< 1��V�<W���.7�=�dv=Aׅ��L�<q>�=/q���
�~� <�ή�/���l�z�h��<�{=w�H<l|$=�_.=�=�`=�����'=|߷=��;#\���u�i��<�v��qO<��N=ŉ�<:`=/h��9 �Lb>H�=�♽G� <v�ż�2=��0=s=�&�:s,���=�:�<�+=��߼4���1	�k��=��<�e;���=�e�<m�1����:���᢬���&<@��=���=ƿ��گ<X�7=U
�<��!=�/w=��;}x���d��#�F3<+�P�&�t �D����lw�� ��z=<�;��p��}><�)��GͼsO=j�5��Z�<��<�*�j~=$��<j�Y�\�H<��R������:�L->HQF���;kD�=`�����}�iz=���f�tu=xC��j�B<�H��?7��u=cI��	:&��<�м����[�鎽���<�d�<���bE=v�=&��=���=������e;?,<��>�@=>���1ڽ�R�:�m6�0���Ґ�C#���6�<����Y8"�oP=�z��뽡}�<����FVܽ��Q=�z=�O�W
=<�;5�>����=FkD��>�:"�̼zN�<�(o�z觽,C��vy>�P��iY=>����1�<*�<���y2����=�PI<S��$oF�G�=��<d>+�=�JػY�=H�"��VI�#^Ѽ��<��˽���;�)/�����<�<�fJ����̿�=Oݿ<�خ:��=M��%dO�@z=��� �9=��<S>�xj�x��=/�=s�=�1<��������;G�=d$=�A(�;��=�#�<q��<Xj=�k��e�9׎�<�d�<�C<_��<(j��!�=�xкػ�Җx�Y�=<r]�<@8༒��;�=�8B�
6��`޼�Z�=�'=q�=:�=u ��&���yh;���9���=R��<�e�=�M=�;�=�>#<jò��A�<�gj=m�<�-����<sS=mϴ=��<��0�ޠ�����?-��팽1�`������;�;C��<�i3=�t��7X ��8W=������� FO���<�a`����u2�=Ft�<�ƚ=��:f�=ur�=�X �)�S��ꏽ+X���=o�?��C��`�<�ٛ<�=�t���뎃=���=�DO=�'=s�>-g=��Ľ�տ={}����<���;�KD=��-�Kz�=�>q�=񌀽he=/=��/�ؼ�	=J�=3� e��2<�-��=l�>Q�s:G�=��
=vo�3�+=��u=�`���似��<�A<=4�+��5��c�=y���ӼD&����=�(�=�<;=�7�<�=����I�;�F�<&��=��1>t�� ��s��ut;{��87m�*��<��'=�I�=%�N����=�Fi���=���=0
l��}	<g>V<�R�=���<�`;\���3;��X<eg��XA>�>L1�=��G/:3��9�,w=�	>Xiy<�n���o��Z�;��w=���=jg=/��=�N+=W����/�*�7�����=t��<2 ý����=��Ž��=���<Nf=��<[�ջ͛����T�.9�<.i�<���<��=�P;�^P=�;Zy�:�N >w�=2�Yp��h~F��D�;�h�`�<r+<آ�=�u�=���<1�:�Ѵ<��=� �=0�=�Qe��M��n�<�Is=�QS=:(�=�����.^>Vw�=�&>1e>��+=6��{��=#�;#����>�w=����q��<��=]�ӽ�?>R�R<�=t=݂>9BT<4tx=�Nj>l�һ��潙��=�A�<���<|��=>��|��<�%Ἴ��<�X>��1>5_�=���;��	=Q���Q����9=4,�>��G> +>��#�y��=iG��A�G�6�=�!q�bD0>�s�=>�K>�ב=_I>X����->)�
>8��=�<�>x���^�����=��=%c=�Ż;���=�̦��R=P��=�><����%�=���=�>=��=��2=�w���<W$�B�=899�)��t=��=��F�s�h�K�<z�&���V����U��<t9A=Tr-�����R�5=K�H:�!=���=�r�=��>3�ż���<��
��V�=`�>G�7�j,��X<{+�=2��=�v;�(�i	�=��<�΀=X�k=�19�C�=;DW�=uA���ƕ����=��?=d)=~���*�(=�o�=�
Z=h$B<ƹ�P�:>��'��hO�c���`��J1�0�r�؏=2\�<9E�����
��Ӭм�
=�9����ؙ�<@ɪ=����k7�$�>�<�[?<Rv��כּ��:��/=*	.=ڮ��;?�<�\-= ��<�b3=@7A���d=L�S�N1�d�= �=@5�����<mP�LE�<��=�δ�v��0�=�6��h>�S\=�р=�C�=�;�=ܠ�����<��u���<�Ϻ�lX<e ý+�=���7�;�ס�����V!M�cT�<٘�=|%�<$\i���<2L� ]�=��
�v(:ҿS�[�O=�bx�����ƒ���=j�׼4�\�B������^�s�=J*~=��E=��̽	�d?�y��<Ȅ6�h�7�����y��(�=�.<�K��<[��=@����E<P̽h|9=�Yl<�A��G}4<�K�1�<��ü��=EE��N=vO�=���μ��w<A�S��e�;��<x�p=��=,�v=��s<9�=��<V�Y�Ju�=���<��m;���;CK=�	�=������U#=���<�ĵ=d�S<�m�=�P���ν�"���U����V���p=�Tۼ|S��r��<˃�=����hz��]����=-��]<%å�e9�=#�4�==�J=>��=��^�6;Sc�І�.��"~��k2<��<�<K��=��z=�3>m�{=O\�K���W�=uM������5���Y=���:���a���<���<}��=������=]�9=hg.=��	�YF�=�����\�=��q�5��<�F�=�k=�ヾ =G7 =���=����<RR�=ə8<c��<�,>?y�=��=��(���=���<�J��K)0=�1�}F�=~�<:6���f�1��<�輲�G��&�jD�=7k=���<�_�=��<�Ţ<�J�=B<�U<<β����=��v=%g<�`:�yt�<���=�U�=[�`=i�n<߃�=�dv�Y�t�u��7���|��i�ǻ������'=�;�<N�D���m����=GЅ��G)=_%��H��=/׬=X�=_]��d�<�7�J罜ʙ�F������ͽx+i=�܉���<�"�����=�U8��U�=̔�<�A��6�=��L�`��=�5=BJ����;SmH������R@=S�<@��Ox��TA-)=�vp=�A�<<9=���=��_=�v�=
�(��OP��7=k�>�4���ߞ=&؉�0�<�<��ȼ�@'���<���=����n�=���=⩩�~MA=����!=ε.:Ue�=�\=�%=Br5�R��<�s��;V��m>U�Ž��=��K��u�;/G9��=�U==�6#���4=e
�<����~�=��,����Ʌ��v����>��q=��)<w=���(�G�y��.��#>�0عr��=i쵼�s=[~=��D��Y=@����E�<}�=���=��'��d�xL �������=uE�=O�'=�f��c�:�m�<�&P=��=�c�"*��x�<�w��s�=���(&�<$�ü�K�B6G=�wA=��<�+S=��7=�+<�@�=�P���߽�)»*¨=:�ͽ�:�%=�81���,��߬��]�<���ՌT��X����~<�=�>��xs1=#��=t��5�,�L9��uT�����'{y=
�#�Ğ��S��`I<N8������q�U�"��=���=��=�.<ݗ=�Y�Mɵ=�ܓ�"�W���>�J"�臤�9����cټ�ۻ#�<�8�1��<�eٽ�8��=�#=A���+�>�>�2�=iBT=g�S<�}��e-�>�Tm��߰��=輳���r�Z�V^ ;�:d��F����i=#��D�i��Ż�JT=���D�������Z%�B�=���;�H'=�;k=|1E�q���g��o¼+�J���=�~���S;��N���=��$<ϣ]�隇=��P>�K<+��<�1�=hX���#�'_�>�}<`Uu��(���`�=0�g������i=�tӼ {�;���<Lg��vQ�j�T=��M���p��j�<��>gd��C��=�Ҽ�W�<��G��o�<��<�t>�a<UŁ��&�<l�<N_S�U�L�p˙=�v=�2��K=��q=+���������l�=��;��� S@<���="�T��3y<�@<~@ߺ%�1��XQ=�i:��D0��
��� �W&�=!λ�s&=QB >l�U�K>�D=�h��N&=� =V�r������=�Y<w+O�߾=�2�<���=v���>j�>q&����=���=�i�<��H:���mo�<�>�K��b�=�oI����Z�=�0e<|μԳ����W=��=���������p=L00=؁��R��<���=�$=�'��뷎��ı�Бl=nI�=��=�>Gg�=�� ��)=�0�+4�(rƼ��8=�O=��K=�J�:��[;��`=a�s=�uƻ�x<U�7��f[=ׇl=��y�d�s��1<<z~Q�a�=mr�=����b���2�.�;i�=�gO�gC=(�=j��=Ⱦ�<h�	�z=�X�<�� >f��;�mm���f=�b]=�e���f;�?��ڼ��ؼEt���"=�*�@���h6=�ڨ;��:@�<Ӡ�=�y��w�;O,�=-=���d����=��/>/�=Q�i<B�=��<n�C��6�=~�=a�y<�m�=a~a��.�=Bq��Ti<��<Vu�=����\�=5#7=�ɼ�Y�<g(=҆$=.��L�;#$����=9"�=��
;<J��`r�=?�=\%4<��輈k�=�%�=�P��L3��X��[�=��<"J;�x4 �D7=�gL����=���=l~��nfV=������=[�D(>��=T��j������= ���{�=���s#:�jz��*��H+>��㼤G}<�7�<t��<7ݰ��詾�.�=���=��=��=^x]��C=����;�<�?=A�=^�$=�-N�$��=��{=QRd�03�=��>bT"=r>�_Y������������=�=b
¼��h���κ�e����<@���mԺXW��sZ�=xh̽z,�����<�<�HX>i�4<�C3��.��0����ߵ��0���Y��H+u��� =翼���<Ǒ�h[�=�#c�ӣֽ8䚼=I��=���p /=Ȗ��e�;=�<�<	��uU�<6�d;&s��/�*���=Dy�R9V��{�=/�=��>��9<���I�=��罻g+=�g�>��e��TB=�S=��*>�������e!�������_=	M-� ��V�{<2�H�i2�����I��<f���O|�=4�S�|m6���<��w�!�z>���p������1����w�R��U��#V����ļƦ��_P5�d��;�$��;oh��X�=s�Q�� ��iʍ=<��)��<�N��!�#�K�:�_r=1f���|Ӽ\�E=��Y��P~�M��1�J�aĸ���<�݃;��>C��=�~�=���=��q;_����>,��=/�<�4=�(�=누�
�=����5�M�P��Ƚ9[ �P���]�A�R��V9��1K�'���}-�b���G�o�&�����K�=�$��Ͻ�zY;�B �&�x�\����^��7���"��K���;�u�4���#��ʲ�仚�_�o��1̻��ֽ��۽��T���%���A�N5K=�=FO���<�u���E��<ZBF�|u��Y�<i�b�9��ڝA��[f��{�w;��i%�g�Ž��ս���Ȥ��5ǽ���<'$�=�U�<nH�\�V=�>d �<{n�;Ͷ��{�I�$��=����7<�=���<��<���+�=]A�����=g�ɼߔ���=��=Ղg=�/=j�� ��H��<bü�Q==I=��=B���=�1�=��=�*��釼F��8d�=��t=A�<;���6w=N_�=���>r=7/����v<x��<��=¼\r=*E=(�=�k=���Q^~=W��=o_�����h��=�ݶ=vcl<�T�=�2>���=��	>��=[40=1SR<�Y0>R�<R<z�S=PV>��=�P�<�7	=+!;����A!�=�Z=�=檽y$�<�P;=�]>��=#]�= �Ƽ/��=�vK<t�;�k�;X�ɽc��;5��=H���x,�=�=�ʁ=P�={:%=��r;N8ؼ*;��T>�b�=[��=��<Uq=R�u��!=➺;�I=Oc>5n�=�c8>`VF=	�'>)fE<�L>�@2>`=���=$�����=I��<���;׮�=��={�;Pȼ<��=�B<�=ތ�=yX�=ǆr=���=��]:xf=���<Ƥ��8����;A�<!;�=������=5�彿��Wf=������ռ�q:�aE�ͳ�<�m���P =I<�<bd ��º=� i=��<��^���(>?bI�M���U�F@	>�,����=����������=H�7�;$߼v@!=��=�< >Y�<����'8��\=M[=ɮ�<�ѯ��1<#����g�;eVA�U�>=�k�=�b�gO�<�_̼���<��;�k=�@Q��`s=�&<o�d�#��=%��<�z�;�d�=!y8<]��=�(r��r�=��<z��=4=-=3s6=g�<�o���<�=?�^=^����=�T�=Jq��t��=5 =�t�����=�Pk;i��;����H��ǽ[���8��]�
=E=z/�;O��<K����iＺ�q�Aٽ;�|G�5� =�3�6�g�Ye�n�}�Ŭ�=��~�����w=g�q����6�9�=:�=*��Y���Xw���F�=�1ؽ~5��b`>��ռv)���=�;��ݖ=C�==�X=�.��*w>](W=:]����< �x>�u�=�к=�?�<~4�=Fw!��<�;B;�zq;�<����&>ܑQ=��1��Žc%	��s�����s��=4p<��*>_դ=�댻4^=��5�<ā<啖<���'	>���9��=�R=d��=K!
��W�=��>R1?��j*> Te�R
�<l�Ż=�P=�:,>oY]��UܼF�.�ܓ�=S�����=��=m�c���[��岽=�ώ<{�]�TVż�}=hD�=SG=���8��<%�<�Kc;<��x=���nɽ�#;�G=N��<J5�=<���Ӽ�٠<>��<uMf=�=������:E��=x�6�BJƺ&��=�_�;P8=ٝĽө�=7= &N={�}<q��n�������]��<ڇ�<c��;��:�1=��==��R=ݟ�;/�x=�Ol���G>�lλ�>>\��3�=LN~���a�o]�����E��m =�Q�>X�A��(��3b�8��=aZ̻�w=�<�⸽T�&���< L���ګ=��5>�5=�����=�9;��y=��;����;�'0�����>���%�=%�=z6e=1r*�	[�<�@��f����=�gV>�P�=[o���T�;���2<d�:.H�<�')>��>�	>U/w>G/�<e��(��=X��=���;q��=_�>�3=~G��
�>��l�qD���������-&6��P�a0ӽ�h����;a� �U���Ko�!S��g��[~6�UQ�<�-����9=���A	�|k<�2��]���I)�3��=��K�1�>����ig?��,��#��Â��:=�V1���Q4�g���J�a��y��ԋ>H�=h��=�&�<�k}�4U	�.&�K��Q��łP�>�=�]�x��~N�<߸U������=�����'��L��6z<�;�V��:=��9k��ٽ����ƴ���%�ܧ,=&ə���=$�Ľ�~�;�9���`��a=^��= ���~��<@[<��;G�����D<&I�,�m�=���=(�<��x�.��=ī'��v=��=�m�=8��=𝉽�=�<�S;�F�<�_�=��� �<���<�ǟ�?���'=���Y֦<й����D=�x�q��k���Kk��:�;�<]E�=y'ԼWjL�`fF<�`�<p<�����!R�˘�=��>�����8�>���q�j>��B1�=�V�=M�ü2�-��=^z�>�e����qq=0~=BI�<H`>4�<�Q��m�;�[s=�|>�q*>(�>y��=��=%,'>�"ȼ�m=5�\=��r=oh`��ew<8z��Zb>Rӄ���L=n(���!�<:CA=e}���XY�Ĉ>Ҝ�>�PT>��Ki�=N�����G����=(�T>��>���>��>	�;/`�<�>
��>�Ç>V��<�%�>g���Ƌ<F"��|�=����ע��]< �����;Z!8=��s�ݛ:ݎ�=g�=�h�=s�-��Pk=Xμ�\:�ᏽ/뛽)<_���h�n�}=�k	=ֲW<�;<��->R4���2=�	�;�<�=n_S;W	�L�<���܌=(���2	����<8 �=����yN�L���*Y�=�.�͈�=��o=o��<T�=���=*���.==Z[�pp�<6#�:a4���_!�o�Ѽ�9����="��g�=�I ��E�<"�9=֏�=�{ϼG=�=)]�=
㼎Fd=�c=`j�=!Ӻˮ�:y>���� >'�v��.�����<�3�<)j���4^=�-�;)�ǽ�N
�I�=�t=��|���`=���<~B��P=�=jB=q��q�<� ���dn<�j���c=n!<�Q�=!�Q=���=
�|�cv��<���=���=�B\=�t�=�y;���;I`$�S#<� �;#3�=��>\pK=�*�=�]��}}����Q:�=W�=���<>�=e�=�������=�<!=M�	�㕡<ͶT=����
j���K���L=�Q<R}7>Y������<9��h�[f=�4�=1�!9���<v��<�f�<~%=�+�=�@4<m�Q��G<�~=	�s�=�FV���:>R=�5�=�Խ;�P=5-=���:���=6+�Fܽ΅�<��<�9�>a��_>�|>q�=��]��u�=e�ɽ��`<��<���=�u�<Z-ؼi��c=���=cց=#,���1���ܼ$W>�<�<��s>Df����'<k���6���^�<򿁼�e��r��=n�D>�"8�T��R/`=�Y<K/P��U=b��=a�,3��!ѼԆ���}�=H P>�-�=���&��=ޕ�<��ߺ��x�D��Hc���ٻ�A7�=�����%<��ݽ���:��t� ����π>�S<>���=>�;�(��Sw��E�P<��|=lI�=��A>�s|>�u�>A@w<aR;�$>�A�=��y=nݣ�Xo�>�W4<��N��s�<��>�1<*�¼��@<���=�!=�Fi�����Cν�8���8=V4!=8t�痏�:cU=82���=?D�=H���^��=��3�m�U�$�<���=�R�=���f���q{��9U;�ܔ�'e�_���߼_8=�ĭ=%��<a��<�%�<��^��l=����T{6�-K�<���y�=���?��<T��f�=!�>$Z�a<<fC�#��=�e�<R7�=@t����O�ꭈ�uT=�"2>{=�>q<�
�=G+��ē=�߼���<��H=�ΰ<�`�=̰�����=C�h�*��=x1=��g�ע��Q�=��;�!=9|b�Z�&�G��=���틙�q��?!=�C�={�4=^�Z�U#����J='��<��8=^樼j$~�ȑ[=�1L�n��;�aE=Ԅb;�P�<7Z<
���v�;e��=�nu<�=u�<x%2=ñ�=�Q����y������ڼ:�d���мԲ��j�=5ġ�}���p�<Y������J="�=��;6�o��=�=w�<�B�<����X�:=�6?=���;�g�=W�<]�=��<.�L�@G:=u:�<T��<!|��|�=h��+=w
=�L[=@��=��:��b�ü�p8����=�;�6���M$=�m(�hڇ<��u�t�E=���=bW�=�5-=�d�:�z��=�k��s0�<?��<��ѼD�	>�N=O�ӽr^���F��a�������`����=ޒ�<l�Ͻ)�I<c�C=%�O�Ҕ	�r���=pU<��<=A��=�@v=��9<��=9ى�3��zG�Vk�=�O�;�d=�޻=8�==�SJ�m>�M�=�(;=�-]���=O3=0bc=�Z0��
�	L=\������J!O�C��_5@<S�=fR�</��˅<���=e�<��=���=҂�<>,B�G1��F��<>�U��4>����v��;���h�e����N=^���1�u��䁽^���d�7�lQ��,~��7<Ѽ�!j=W�Ž�O�E�ռ�ͼ<�<c΋=*"�=���y��i=l�!=FE������ �	�	�<����FN�D�Q<��=2�>YĽ9�=�8;N6��������`�j+�<=.�<���=*���� =#z=$dɼ�}=�瞽Y>B�M���#>p��=���<-ލ=!��=�g�=�෼<�=KIi=�̼�샼��<0t*<��Y��e�<@'ԼR�="��<��E<��I=L���$��ג����eth��Z<.X;zB�<ER�=�@^=R�G>��=���>�����m�"�C���,=��=�߽�;
<R��=�_�>z�q=N<
R?��ុ+�h��=��E=�",=|������=��<H��=a���#N�=K�0��K�=<�ռmP�=�V�R1Ҽ�����<A��=6~R=�m�=���=��=��=h����#=��Q=1�U>����:�������7��=��F���r=ۓj>�XL>��>�>��S� >�+y>{�=ﻦ=��=�t�>G+���x<�Sz=X�<o|�G��=L���S�N=���=��=�R��@E�J�G=��|� �;9��
Q<޼:o<z<��N�_>�=d�8=�`=�<�=��?<J��=c|�<�Em<c=��=���<ϭC<g�����=�S���93=���=wE>�D=���=�>O=%�;j燽���<��< ��=j�=��<.=b���#=и >=h=Y���H�&���׽��;:U<mژ=�׾�]<<��=�rc=�F��Uu㼀(=δ�=�f=#�=F���5���N�=K=�!9=&ח�r=��=u2@��"м��=_�<�ֽ�s>�����<AB�=���=���=>��<S����=��t=�MJ=�7��%g�
��$�7=X�=gW7=�?=&Y$���&;�e�=��B=sU=Fr���޷�=c��=W�	=ûn�D��=F�>�V�<�\�=韭:nT����;�<�a=�����ؼj��=o*k=g�b=��g�9�>kv��Eҏ��=��C�<F�=;T=\��<$��ɋ��?ݡ<��=�1f�˘l�55=�C�:�v��L`�f�b=� ��_�=ԡ�wY=a>n�==)������=<��=ml�=Ho�.=�ɼ�B3<|[�;`-m=�L�=��%������=�:D�噙�3�����=�R½p9��S�<��`��q<=r�b=�m�R邼�';��p��<��Hش�R�;�_�lm==�_��}=��";���<pd]="}���T�r��P��=Ϡ>�e�=/w8>��=q��w>���=U�M=�[o�ý���<Yl*>i`�-
�<�Fb��;����@��bY�}�&<i����tS=���=m��<����i���t�H���</�=!s|�bP=�(%�P��<�W�<���=7e=���=�=z�=��=���i����=b�k>,?�=���N�����=Z���"]�z���}(=�]�=�<>tә=/@�=XZĻ��6==��=���=J�>���:�0.=RJJ=�S=���<��=�n�����ڷ=���a=B=��>yï��"�:�������ȗM=r5H�V�V=���=��f� [ �v)����=P�༉@=Gp�<��;��$=Tw�<~�v�=��M�g��<��g� >0E�=�d<�c��H=T5L<�g���P�L��=ԑd��e�=�2��X��<�ٻ�\(=)4�=�U==V{=<�=��<k�=�t�mة=����h<6u�;���=Vi�<�Q��.ܭ<#���k<���;�w=�in<גA=�=�Ϙ�)����<�Ʉ=�Z��sɎ=n;�� =��=w�#��)=��s=3�<T��=-�=p2n��o	<r��=I�;�b輡����a7�xw=7݉<�q��� ^<��<it�=C�q=@"�<� �=V3����-��PY=OP=���=��[���)=�����r =��=~�;ϗ׽����.q�=b���,����<~�7=�Y��^ã���<�q�=�B=l�Լ���<踒�J~=�ܸ<7��<QM�Wɦ��(=��U�2��<�"���ؽ���=�^��%gܼ�/�;�G$=��D=�\=��=�#���<V`�=��=�����s>��9�(Y�={^q=Ui=)���'�=�����b���e.�*�ӽ(8�=ʈ=SS=O���A=67=bR6�a�2��/=Ȝ]>��=O����<����}n3=��-�����<����t�=���<&�>�>��g�<w�P<�T�R�Y>�I��5	���e=Zٻ!�"��f��=ъ�z�}����W
��*��;����<���"2�{a�<��`��.��,�U=2��=�i�7�`��e>�R𼻽���4=): �^=� =ɸ=QH=+N�D���%=�Z�=Y
\�2~}�[8S;�n�+���^6>�����t�Ec�=��Ss�<�y�NY=36�=4pl���<���u�<x；�׼�3�=ޜ@=�-�<5��<�9�<cM�<Ǌ��B&@�/��n�<k+�;��g<!ʺ��'�$<ʭ$�f������j����.����*�!&��!��>A>�]�7���=e�=:m��U6��?D�;pD=?�h�K�uH'=�/|=��=�K��#.=� =8��?B�Sm�<s���ʝ����<?�O=�@=�WW���<f��=�~��*�*=MѤ��������<0]���Ƽ���=Cp��V��=
h�<��ս�&3=V�=��/�iG=�R7=�.ʽۙ=侻��0=�.=.�=p�tJ�<&��;@�@�n�U����=D5��l�q#ƽ�H��6��;_��=e�����.�6�&.��A'=�[`�ӈ˽�n���4�,�<�����M��42�<�~�=r��<��9Ē���4�!y�=Ff̼jԺ7��@z<���=��ۼV��B/�<�;=뻗=v�<=���=;���pO��p]�=�}���=��R����>Y.=!�]�b�体⇽�cＯ�f��-���=�
�����*=j��<!|�B#A;D&�=��=Э*�f�|>[<�;�]�<C�X=�h[;6�=��=Ze�=���=�2���ӽ?=���=k�>xۺ=:[�<��<9�=�Ύ�3���Cbc=�Y,>R�= ^�=r�}=;�;��<���=��)<!j�=�ȸ=��P=�ת=Ɩ�=�����z<!�,����*�&���爯=8�7=x��=� ����=��r=�)�;���=�[���<(�ƽ-w�=���<d"�=L��[�M�L�E>��E< O=���Rc=��H�cɀ�p����<��,=*�\;���<�ƚ=�r�<��<�6����
�f`[�%����<տD<̺k>D��==1Ｕ�|��.L>��K=c�U��Ņ>��>���� �Dɵ=x�8=�J���E�=���3h���(�����E:=T�h��Iɽ?�<�v3=7ؼxY�=��V��ż�)�>�L#>��=�t=�7D/=t�8<76��<�z���ʠ�������>H���ԨT=[��=Z?=g���;��2K<!g>�/:*��>�\=TZx�(G�|Vv=2�.=��<����6�=�Ϲ>�cD�=*=N�4�x�s�~����"����`�	��詽<�8<�޻K=m��=<O��=j&=[��=.���}�<UӇ��Z��ɡ=H��<�,�=�N�=�B�{���Hۍ=i黒`R�Ȁ�<�y>\Mc>S4<�j<׍��O�<��=��v�y�=�PG>:�_>�Ӂ>LI���I�<�f�=PЗ=�E>@Gk�1�>�[�<��<�E̼�t�_
�;���`��=-*���=�=�[�=�.����?J�=o�;�f=G�V�,���8�7z�CY�=YP*��{u����)j�jX�=�~-���N=^4b=��󱽐ɘ��W�=ZQ=M�P�]ܒ��"{<�_����=1�Ӹ�<�(=Y�T=���=�Л<��l�
M�;3�=�=+���7�w=��=i����\����8�:��ü�p�� ��:e�	>�Z�A͒=	r�<W^�=�mm=^�%�5W$�,R>+���;��>�h���=�o��f�,<j�J�	M�5�=v�<���>@��t���7�ܼ�X��]S�<�h�=�o���B�[_ǽG�W=�;��>C�>~�e�j�'<?׮;�\�������=��������
<e�#�h�=W6U=�G<Mm��p�|���5���E�e�+=r�=>�+S>���<�z=�%��Ӽ�!��U�=�>j�>�P?>��>�h�2�k���=4$>�.>4P��$�>��E��N��Y�t���=K/�����:�!=ؤ���~;��V=՜��6�}� =�֛=�"����A�"g=<�==��V��B$�ʄݻw7��}�!�.\�=�=�=�����@�h=wJ*�e�N�����j��=����SR���h�ڃ.�݄=ۖ<x�=�"=U��=i����bj�ݻZ=,v>��/:<.�����=Ǆ=H�g=�p<j�=�3w<҅�<Ǭ�=�J��x?A�@������MZ=K�p=�}=��<6���F&�c�+=|��{&�X8<|[B:�\%=8+�<��e�z>7<��(<o+�<���=��=��A��?�<���<�D�t�{=uDE=��˻Z��=��u<Ʌ�<��v����=�b����=��ǽ�>�<�ی=�L=�".=Ӓ��iJ=σ==.�=��<���;G�=3�B=B���E#��ge=r��=@x�=N���h�=�8���]�='>C����<�,�=��<��⼅�=�:<�>6�	�3=�ۼ<��\=�[�=Z�żl"��+=�W�=��
�H">��=n0B��A�<�_y�}�=��n�=��Y=��=��[=Z�Cۗ=�Է�{.=�W�=����N^<CXf�#}�=R]6;B;>mƒ=N�żfƴ=r~=ϔ��5�<M��=	��=��������:č�<*��=�����;=�*����<�<�DV�m��=�mT�"(ۼ�>6�+=O�@��!��V��+>�@�_�)*��g�~��7��O�޾2<��sX����4<iѼ�/�=o�"��<�]<�^_=G(�=ʩ�<(�X=��=׌<Í�<�!�=�#=/4� ��<�G��1[<�J�<��m=���}�B�'=-�O�);�c=���<7Iü�1^=$�<x��<S�J���=��̼����5$�KH�3��=3� =��=O�5����<!�
���������;d�<��I���5=���=C��<f!���'����o=�`���s�<��=�G&=h�<Ak=�7�<ap����<I�;[��=/�*=��!>�J@=zj>>��=́����	���i�:��=۹ļ 6a�ؖ�=CL�=�H)��7=鞤=�3=��1��1��Tv<� �9@�'���M���=��=�}�<	���/���.�t`;�Ӕ=�;�=4J���=��%���Y����%����=��b=��C=�t�<O�ӽ=wi<+l?=:+*�Z,2=����x�<�V��LM���]=�A=C5�=���=���=�u�=��<�V�=Jy�<�=�f{=q��=H&5��T�<a�<<�Ճ=)�Q�I=x���\;S�G:Sg�<���y��ڼ�Ȑ�
�=9�������7�/�QH�=�ɻ._�=��$���/=�=v�(=�<D=��ʼuԃ=��T��ȁ<<��=�E�="�{��j�����=Z��V��� �+��ؼ��ļ���<�[�=z3��[\���T=l�ؽQW>�����*=�4����{=��C��K���Ql�j�m;Hj��}�=-ƻu=j����.=˗J�J3�=������#���O���&���<m<җ]=��<�t�]d �L6=�2d�&A�<��[=��~<U�B��=�=�=��`��r�A��:��<n/�&�6��=�'
��;>}[y��XA����=7k����>�<�Li<�\]<�����5=�U=+��]�7=A�F����O=5(a��=	��<	щ;����OjS=���C�V�J�Y<���7����D����/"=�}\=Ϫ�<�d�=�0�;����5��?���6d<:�=3�=��=Go�=�t=R��<��=��#=��=M�;���<�� =H�<���kX<@�n=��P��?<���{���=f�=��Ի�&^=K)ν2�=��T={�=��=�:�<�KU=otr�����gQ�<�� :�i	<�DL=�{=L�9<�K�6i���G<��%>�e;�`a���<S%=K4�<#i=�J=�!�= z�<ը=��&>�ބ��=ʐs�$���2;9�=6��=�������=/e��Z�<�����M���x<�0ּ+r�j�{�1b���VV����<�e�Ȝ�oF��H�t��V�=B�M;��=W�=�˄=����G*H��r�<Elw�xս�ǼL���ӽB@w��K��OP�ð;<,$�<�T�k� =���=W� ���M=�Q,=�1�<P �{�=�3����(<��P;I�=w�&&=��̼��>��;h�R=ܪ��A=�}��(BN�s(�= � =i���M�w<xG@=%Dغ�=���o�=��>tգ<�l�<��=BM�F��="�)�
S�!��<�<p���ʺ�<��1��2=��?=E�ܽ�H����=���<(-/=G���G��=ϐ׼�����O=*�ѻ�Gc<1u��c�<��U=m�9=��<(���:=�u�=w7d�(�	=Ǌ�;W�B=�y��{h@�+��<�x��j�ѯ�<�v�<'��=� �[>eC=�����ͯ=���b��;���=%��=��A�̰�;n7��rb%�����������"=l	X=��==@G�<St^��28=���<�={F�;�����d�=�V=�y=|IY<I��3D�;��i��Bu��E/=��<m�0�d&=nǻPT�=7�@��}�<�h�;iJK=)��<���<ͻ� +�<���<l��Q�=�X�ջ�=uG����<Y�<�z��c?#����k�=�5�;0Ҙ;6 �T7�=���<��g<s7N=T�N=�(�l�=Ӻ�<�Ev�a�<��[=�\�<�׹<�7��71�=
��=+�<�p��jI=�Ύ����=�d��On����<��j=\n�<�7	�ګ�=y�_<"3=f�M�y�<:Cݽ�ً=lِ�Q�k=�)�՜L=� =��=��:��=��=�/�x�<mg���\�>�>��d�܇�<�5<��i=|�B�#Bi�.l=���=�̍�z&��~*�C�e<�K�=#GE=��`�S�J=�AE<����ּI�<��P=�+�<HŅ=O�<�W������@<��Pl-=������=x�r�%}⽊㜽�^���Q�==y�<T�R=1�ս�s���t<�Wj��;��j���W=mz�;y�<�+>�I>�������p>�_�=޵G��>��>F4X�I"!=�\�=|{�=�̶��@<�ռDּW��Մ��cٌ=���=���<�d= ���($�<�xI=ɯA�jCɽ�u�>]K�<�U=HJ=F�'��<]����V;Ғ>_3׽W�L�38-�=�>��v���>Xm�= ���_ =�n�A;�<x��=�C=~Z"�zg	>R�<슰����<��=�Fx:!�>���=4=#*��W!��;t�<�$!=^>A��=���������-��A֠<TV�=G�����#<� <���=����/�9l�=���<�;�fT=��<AU; �=�#��@=|Z�=�?<t��< T�ը=,��=�k=0ӝ=�d�=�v�<��=N�#=n��<�e���D~��f�=(��<�i�<���=��ּU�<s��;�é=�>���<�;"="R�::��=?3�=��>��<=�T���P[;N�@�\�<2��<�Ԥ<r]�S��;���!��;�i�=|.�<vVw;�缉1<�%=�F������-�>���=�O�R�<=�����l<�͟=0fۻ�	=A���~�3=_H=�k=�j\=Q<��<�T��=ѥ���F=8OK��Υ<�j<�K�=)üÒ��ͪ=kb���oN���*<�r�<0/�;qѝ=7�=a�;����#<�X��.��s:Fh<��;�<Ip=�ޒ�:��=Y��<�$ ��p���(�<��V��=[��;�5���Ө:j���#�����U���q��M�<�4޼�n�=Qh���dB�~�<R=b!.=pLF=.�<(�2us��y�;�Ѽ��y;��=0�<<�ܽ�;<��<_}C>C�<��=\��R�<����*쁽�d�<�0r����<D�=��8��B�<�ʼ��p���
�����'	���w=n�<m�d=�5={���]m1�?B�<�:���=����>���=0�=�p�<�a�;��,��V>$" ��m\=i��=��;�⎽:d)���;��j��� <��¼�k�:��-���Y�j� =�S����<T��;�֐����xE��2=T� =�?����;ř�߉� ���黋I=�&���-�< ��_���;;X!�=�D��[��=��;�P</�H�M�-���}��:'�^"w=5�>u�X=��Q<������[<�#<B ݼ��<H��=�ɓ�/=�<��<��<�D|<*�O=Sn!���ź.�-����<,�~=rd��?z������^���+�N��a=h�=�V=Oٺ��Q=㭄��=Ѩ7:��:�&f>�[T=-�AO�<�<;v��<CC��p<d���� �����<O��=��;f �;S�I=��=���<y#���=�ބ�b?�=���<>��=)�=d��"1j�4i~��b��~@<��/<U(=<�=�;��<��ˏ�<�jͼ��M=�#���8j=�얽�h=�i,<|�(��
A��q=�b@�� �=5N�=�n���������<�=��=j% =�==�v=������D=<��<PM �b��ҹ��������?����;�=�r�;��@/���^���c=��-:�
=��-�R�q=��_���Z<�G�<F�q�ދ�=}���P�S���d��<��=�8\=2�E=gy�;��=*�S=5�<h�u=��=���=��=F�����5��H�<e���y=t�=T̸=�,=��j;���`ۏ<�$NN�A�ųy�%�=��E�Aa�=�ѽ2����o<,�<�<Y�hݼ�z%=��=UA,=�=�4=��=�+9=`o�<�/�<.�> ��=;EG=U>2:��ǽ ?��zƻ�Fٽ=�<�b��o�=>0E=����!�l<��=��*>�����y��~���Y!>g.=yt�E�@>�!= Iս뚽��������;�)�L��f�=U�_=봘<���<�Ս���=����,�r�<6��=n�=?>_=��F=����i����<ON�=��&<G6��d�=c�>�4o=�����B=�u�=xБ�?=��)=�ҡ<�ث:-�=囓=׶8��*y=;��=O,���b�z����ɳ=	��<I	�=�>��:��;;{�����N?=��=Dk�=E! ;��'�s=��=T���ɽ�-m������{=�y�$ ~����<����u��;F�<M���M��H.;ɿ�;�G<MRP=6�e=�㝽�3����>� =J�>:f:\�l:-<�q=$l�=������=��<��>�2�D�,=�� �D����~�,��<�U�;qYȼԓ��P��<C�[��Jf=�dN>��=1�B���>��>�U���R_��pR<�n����ʼ��<�
�=/<�zN=�H�<*�;/�!��r:�9T=�gw>m|�=sb=�H����}��<�4�<�@�<9�>IE;>a�3>�U_>ު@�d�=σ=��5>���=���.�>^%�<�K�=��d<<�=4=D��Tl���h0ʼ&m���w_��"y=t�=3�<+dL��؍�2�ӻ��=���wY;�!;��L-�=��=�͆��W���H�<��=�b:=qo�<=(L=��"�1�(=��<LT��֏^�7�x�P�[����6��=����ܷ<]�=��ҽ7z*��&=�}|=�!x�w@�ǒ=�ʟ=��<l×<�}=����7=ϱ�=�.=hf<`�;=)'�=%�>^��=��=���L!�:��;���=�Ŗ<�~(=����q6=����.X=.c����<@Qt=�#���<�=W�㛿=�R6=�(=� >��=&�0=Q� �y���~=Ɇ�<$��=�m-�i.%���=%?����A<4Bɼ������N���=S�:oo�=ڮ�=l�Q�ɻ����?����,��=8<��T���I�0�/�`<c��<�-�<��=)�	�����w�<��+������o;�s�#u�=��7����=�K�Ǵ�<<�$=Ғ��/�=ҔV<Ҵ=������߻U=�_=�~��Q�F��<�;��G�=�c<����?��X4U��5��g>����=��:ZP�=G׻�# �=�8�=,��=���:��</Dy=\a�=�ĉ=�M�=���<���==�<ӎ�=�=�ː=�A=��������3=2�E(�=�Ǒ=�z��)q=�y=*r��-h����	����<�;�<4�<:^�V~=6��=z۝<�D�=�]��.6=�U=)��32=�D����<JS�Xu{�+U<�����%����F=cB�|���?�H=}������k���=<��=Τa�ڶ`<�>�/>���p=��=(�=����=���!�ܬ:��0=�D�=��p��+�<<������V�;>u��<�Q�<���=���=g���5�=�$x<b#T=@7�<���;
����ź?�/�6�=P.u=`e]�=���N@�=Q��F	@�	=f56=�Ĭ��.�<�a>���>t�=�>?MR=���j���� m=��s�wf���*�<�-�B��>��"���0���<��q���~�ڎq=�v=�U���	��;��<�j�<&�=�y<]����Y=(�=e��g��<��S�CA�;ܙ-=w;�=�R��=EZ�=�陻Y'���8T;}������=�P�>:�D>��=��9<ڰ��̓n=D�y-�=�(>dm>�T >t� >R�n=�D=���<�N~<(9λ$GX�@D�>Yꂼ����A�>��=tc�>�U=ݧB��d�:36.=�<�+�='��� ���R��>��z<�b�<D��=~�5=Eq`;�<�=��=��!�S��^.>�ˏ��@���Z>�٩=�BI=��u=[�<��7��*�=>
��RVF��۹�#2��Fh>��}=�r�=z`�=9�S<!ǃ=�Ͻ}��=/�>�@>�lD=1�=�QҼY�>~���=��>��>iќ>v��>��<�(�=�w<>k>�zI>u��=B��>s�1�@�/�.�4=^���=��׽�P��E����.���0��}��=�����Z�=���ϳ@<p�����f/c=�RɽY�h�=)���1�2ǽ�a ��$�/�(��U�ݞ����u�(;�=���I�#�к����սs��=H���[�!L���"�m���1��ʅ�GY����i���%>�ݽz܂������X���=嶠�����fu;A����#<7E>��<�e��X>l`1�/<W�1��� =������x=?�<��F�U����E�<��B>w�\Oû]̡=2[=���<�$��xҶ=��=pY�̴�<3��<w��������<��=�v
�[x;Z֊=�S�;�D;��=o�=�E`=���ZgĹB�м������;�:��;7Q��[$>��"��
�=��=������'�� ��BR==º���0֕�йռ|s�=���=�4�C]�R���LW;��;���<玅=���<�2��=�:=�#=o�=B
�����s#<��H��kڽ� ����=��<�>�zI=�����Ľ��=�=�<�{=��V=MJ�=�.<��>��T����=sIf��w�<�m�=��p<�>�Ұ=?-=p'�<d<>��z:"\=A��햼��=����_�&���=�:>HG�;(�U�>�IU��|>|�6;�E*���=P�pD�H3	>� ��B�;���F?�_%����<d^=r���y|>�ty�"p�=�� >U���*�A�ːi=^�(<5A�<b^�������/!<%w+=n��=2�=��p�u*==��=r]���}�=3_�;��=�]�=Ñ:=�����=_ǽ"�=��<Io=!� <��={Е<ELn�rqS=�5���2�<��=` ���)��R�Ĝo=��=��	=�㒼����5=���?4
�q�<�'x��|=(�=;�d��U�e�<�J�<#(�<�ڊ�=x�)��A0�W2����=��o���<�������;sT���	<�*F==n��
�pk���{�=�?����!�c;|��=	2�iu�=m�̽��<d�˼����<?>���=�No:�T�=h%m����=�#�9�=V=�8�=��j�ǲ>�$V�k�=�FR<"�=�߽#-�=�*���=1fռ��ļSߎ<|�=Y	0�%6�=�,+>g�2=�����8e=�}=��<N	���K<Gȏ<��@=�¶=�嶽�'�;=-"<��=s~0�+`��~��D���b�=���m�;6lt=@�<�W��zqu�|^c=�V��C��}N�=~�!��qO;�!ڼ�[n�j&��<~����8=��*=I�H�7��>��<.�u� ��5��=V9;��;ii�<=��=-�ɽ�=/�69__߽�F=7�=���;X���3�<,����H��2�^���2�W<>�
�<��L>��@=iݕ�u~>^-=Fa<��=]v)�D��;k�E=я>��=�0���T��W��<i�=N@�>�^=|h��&�����^�-�ǐ�=S������=p/�!ȩ=�Ż�tc�q=�=J��<���l���=�{�1� =�51���/�~=��*>mW�=�4�<�K�<y[�<��=�H�<����d=b@�=��Z<�ߛ�T�e<�0�<A��=��[��X�=Pi�=�>Z=��B;�ޥ<m_=Š�ŉʼ��b�����:ғ<�z��$1�;�!�:s<�<w��=�5Z��ǧ�t��C�������mн�s�=.�c=��;����}۽jQ=�νa�y��B��<�=j����ڽ=CU���7>2����߭=���=��[<�Tռ���\\�=� �W�=ngf�_L�<��S>6ѥ��/9='�
=!�X=�1=�ǩ�k��=2D>��ü��1�H�c>7!����=���<�q�,��Q3<�����>0K��ܤ�=��3�e?>=� ���l��=�#3<��g>{��=�X�<���=�iO��ii�Ke=r��:N=��-=�"�;��=l|@>��W�f>�@Q>��6=e�<>e�=�Ͻ~��=�$��S�=������Ž�����<v*��oȼӔ߽I��%�2>�ӽ�~�?e�=5�7 <c`��͖m��9��S��ސ���!����UA����	�GM����al����b� �H,0<k�e��|�<'N��_���I��:iq�=�����O���ڽ��彁u��Ӌ<8�?�	B/��8��bv�%�
=uVq�қ	����9琴=��=@\>YX�<�C����=��( @=���������v��n$=�y�=iܪ;<����|7��gU=�=���<;�!�e���Z��=���<9�=���={�V����=��=��5=��f= _�� �@=Q1=�8Ž���:X��;6>�@<�=%���q�W����U��=,}��7.M��CX=y�^|��]x�As��1 3��
���p�n >�D�����>v9=i=�G�����oX-;�;�R�=��=6 ��j���f�9���=���<,!>��4�t���� �<m�;���B<��6�,��>� �=M\�>G�<.>��d={��<�W>�Z�7��n�>JϤ>x�=�;4=�d>�j�=��>��<�%>�����1>$�>mC �*�j>�M�>��=wW<�uX>�@�P|	>
��:rв=|�伝�$�5f->�R�>vҒ=\A�><(�=C<>�Ɗ�-�>��=��>��>b�=G!=Ě>��<�T=�#��D�Ǆ�>M+�>���>��'<���>3ҩ���>>��ʽ�w�>�Gd�f �=	�<���=�鯽�/!���7��m�Fw�=��N��ҋ��&�=��=e�ǽ&=�T�<6��=ģ=��Y;�4M=�S���M���=���� �=Xa����=���=�v���<�ɪ=0�)���>�>;��< K�����h=�Z�+����Z��D���d�3g>Z��=%D<C�=7�Y�+���U;��<x�=EE^;v�;�%=��z�-N���'�ۀ�=��%�޽�����Ҩ�]l罸]~:��s���N�cS���vh;�j�@��=3b<Ʉݻ_�>���=|In�|�U=�E�=s�<��~1=[O��>��=�Bd�-�}=�SI�8,d��D�F�:<< =�@x=��=R�E���=� =}�<N0b:Q!	�LH`<�tP=����e>L�>@��= ��d<�<(��7�#:��=���?S�;/��<G���V��=�m�;��<�A��㊟���	<[�=�&��E���h���=���>k=��V��>üG��<R�=w�y����%�V�䙥��~_�K�p��+�<�=�|�m)A����<�C�@�<;�9�%u<*�μ�~W=]��<���=����;`�=Q�=4.=�F�<��B=��<�����=Խ4��=�&�n�=%�E�J������<���=��4=�$�=�+�<�����ܼw�ҽ긼Μս�Y=޾�����;�ً=�؇=��2��橼ͪ�0�=j���/���W����1S���z����y<�-�=�'�Ld=듇����=5�� M�=�,O<dyZ=5�~����=Zԡ=*�=�x�<���<�Mg=��_=���=��<[�=�:N>��=!�.> �={N�=?��=Y�=��}=]cq>�Fz�u)���ʳ=�]���7=�@��\ш�Ð"�{>��^��`b>QO)=N`�=�7 ��ύ=ƒ*=تy��n�;oA�oM>50=���=3�=�	�������<	����=b"�;�C�= d=��>�3��>W>����Z�<��<�C=��Ƚ<O`��&����<��+=`c�����5~������p׽�纼܂�b��$3�=���Ci�=М�=?"=���4�;d|�=�r�=��5='|=�8>X�L����;^Y�Jd���?�/'��~Ȣ;���{�<<��=����H�=)��<��=���S�;�e�<#4�=d�/�A��=�)�=>���	)=� >�]:�\�����=�H��.��\i��CU=K=�<}=�������8��<��V���<چ��EcL<) �������	���B�#��<�m==�ic=Q��=�l��O�=�ؘ�����0c	�I�ջ�?ڼ�(�<ԕ��Ԅ<]#Ž�p=5��=�:�<=~���)/��ҝ���X�I<Ǽu�[��=wp�=
"�xdU=�	���@<�g�<b�<`�g=�C;ӠC���� aM�g눽�e}�B%��[;R;��%����	�<`�^=w��~#���X�6N��B���U&<C������<�';�$i�kjS=�՛�S�[�K=�۽υ���oM�E���l=�zN�3� =&��o�ļK��cN&=�0ƺzo�;�}�$��=�n��s'��� =��>��;���=�)�����= I�<m�;�L����;�I��R��Y�55�=��H=n���Q�>�����9=�/2����;��=�9�<����(h�=���=T��6͛������	��9<�x=s]#�f9���*սЉ��Xf��CS����E����B��=���;�S��8Q<�����=�(,����:|��;B
��x�=Ge���Z�=��;���<�߽��;�O�<��==t���&>��@=�L=L�z�M=K�=�u�=��o<��};��*=^��P����=d�S�y�<�vX<���;�k,�!�����:5� j��&D����=E�y��c:=ں�<�3�|�6<�[�58��<S=d?K<���=V�=�����5����<���[�<q�]������P��D��3X����=���X�=�E=���=عT=�6U����<�>=�D�=�[ü�ֆ;m�3=q�H���=Fr��'�=��%=NA=�҂=Q�ýk?=����9����=���$�w��<�p�����<�7�sټ7w0>�X[��*9���y�������(�Ƚ��(<K�|;N&2��@��$�S>��w��=�S> �к�2���=䚂=�tͽ)F�����;��:���=�\�;3�K<�����(������cͻ��=t;%����G��=��=F��+�=ѻ�=���=O"���=���=����c�A=IZ=V_>��\</
�*H�=��<<�h�>Q�E=E�=�F��U.=��8�z�=(��>?�<��?�8�J>'�u=1��=U�>%���I�s���+[�<9��>+�y�Bp=��<W	=��=Kp'=�0_=�R[=�,�>Y�5>�$_�[��=o�ӽo}�=��>{��=NJ�=��0>�V�=���ڃ�=�~�<��>���=���=�7E>��:����=�,Z�B\�=��"�~a=	]�;�q���K<^v��_ļ���<�ҽ��Y���D���t�(<C�Q��<���=�"�<�������`��=60Ѽ47�=.��=s��=~�����[���$������-=h�1;K�ȼI]�<�w=B�=B�:�<��-]L�oW6��yf�ҡ�=��:�}��DZ=��=�/�<QD�=B������m91=���<|F��(���:�?�X�xԪ��T"=I�ɼ|貽�>�<�ս�L��<9= 鬼X��<洽~��=�Լ=m6=��	�;=8���������B���9>���9�,�< �D=ęW=R9���I�=�F��k��;w!�;���<����WՃ�ĭ==��;�`@=ښ���"7�����Í=C��� �0��<��U�o��=i� ��$��L��=2�<B�?��Np= 1�<��<TI�<ݽ~�c��/=�*��s�=�gD�Mx�?���Q�F<����-�q=;�G�A���O@�rt;=y��8�=��<��i=�_ؽy_ʽ�Ǌ��Z=��+��:�=܏��g��\D$=��>
�ͽ�V���;��@=��U=��>&�=�=�ж:�=�J;Φ�=zG*=h
>��>Nb���f��U7�&Yʽ��&=H0/<��*���%<����#��6�]�Vܰ�d
�=_��<|%<��)=	�8�|�R�k�N>�Ծ<z[=��
���B=�!<=�h%�&��=Cek�#}-��;�2c��U��=r+����K�������4���</���(��<�6�f�=��=�X=���=E=87)=��w�O=���<�i��܅<;��;C9�<Y(�=\/���r�=g/3�W�=�W�<�)=�p0��=j�-=��<�Z=+�x���C=��<=��<SFn��Y#=��=�/>�����=�{q=��ü�`����,=���<�����i=;\�;��?=�͆=p)2����<	3�<<؝��2=.�-=�P��$���=�B���t>lpH=vI�=:�<Ss�QQ�=X!Y����<�ɽv�A>�����p�弥PʼU8�P���2=��>�=�NV|��X�=&�=E<dV��ܴ<��=�];[��=�D�<~��]e�=:������=�lE���;��->��<ϫ>�F_��9��O��&��pL^�|+U=�/�<�^���|=.zV���<m.p���}��H���<kT;���I�=x�½����ڏ��_Ľ��8=Ϙ�0�0�_�D=sړ��t;�2�V��(�$�,<�ּ����=���2T+=0k�;����� ��۷�7f��5C^<�;�<-�ս�o=V�����<�;=�=�9:�2�=|0����j=A<�W�=��W�sPM<�}�=�Lv=�'���I=t[��~�)=��;`�
=��,��2�����</�Լ�n_=>�A=6�k>(�0�lc�L��=d�>>!�=k���������`>��G�=1�T�Y��;)�A��� ;���F�-�P�����$��=7��*��K!7=���xY���<���=���=+-�<�"=|�*��˕<b,=�29��P��5��rz��Bl�=���
�L>֑|=_FP���F��=�	w=c�g=��=ξ	>�&�=b�C=��)=ͩ��OJ���0���=s|�?�	�K�`=rU=7O+�d
�<JP��Y���,O
=��s>Wd�=����>\�b=��<w��������s>��Ek�>B^��Jq�f2�������>��'>-F|�F͗�>�m���=��<;���(��h{#�-��<-f�=�������.�ټ�)��fR=
�t�hk�|�=;�<J���$M;=I��=�LǼ�$5���n���=OkR=�.4=�p��}�U����X ��B���`=`b>���<�ʽ�I%��_Z�ex=��;�Z�<B/��zAy�V9۽�o�	 9�a��=�м���<Q=孛�J��;��<�J<0Q�=�3s<�qv�����A^�j�=Jhd�R�=nԒ�6�<<1���������H����<�Iϼ;��<9 =�P�=xc�='�'=r�;=�ǼIC�<������=^��F&�GA�=��;��=x�;߽�=at�=�I�<��+�^<��V�;���9Ӎ=��=͓v��0=Ьn�J?f�'�:=���<W�<��<1�:���+ת;۷<G����db=YQ�<�P��#{,���z<��J=7f�;/�(��<����P�ř���I
�_*J=�z:�x�P<IK!�W:=�LF�e�<�׼.�;�y޽�c��D�g=�Ud=��<��=�tK<Jm�=Hc�� C<�2�">b<-p	=�;�K�;=x��
:�ɘ=˻��(�i��e��<l��<�oK<oK�= z�<~�N��� ;�8���v>=@�|=kN�=�|5��O;YL/�h�Z=����k�=���:y��<�(���i<��<`�l��l�<��(���#<Ԫ�<G=���X=L��<�7�=���=F���⦻<Z 9�āN�B��=Q�;sa<�z�Y=>��rä<Ucռ���z�=�p����<�'�q�6:JG���<�T�tu_���<S�;��:gM^�Iz�=8z-�:Da=���=7�>��>"��=L�7�+m�=�7�<Pct=w���Ғ�<,��=� =�]<���c�ߑ��T=i7�;Ma��J�<4�=_P6��P|=椖�_��=��<�p���*>Hž=��"�#� >c>A>����>=����<�6�=k�=� x�p)���F��1]=�_{����=�˨<�����G=�i�Z��=G�?��7�=�0½��O<����#�;�;�=���}M,�,-�;��=!S���۽��`�g:��\�<��Q�I�">b5m=yO�=B�����<�(5<\��N�=gB$>�4̷<b	�=6,R�PI�!%�����5?=�f��Ū�<A�������ll �~L=P�ދ_<^�>�@ܽ�G�<����V���]V�;���V��J��<��+=���>� �UZt�_`==Z,�e�����>c�gY<�������=�73<�Bq���#>�!$��!)>��P>���li�=��= s�;�7=�g=��>�s�T[��x<�ܼ��>ނ�=�����j�FU�=Nƪ=����/K=|y�>���=�����K}=����c!=-ei<�Y=�B����	�|�6���>�)=O�>f\޽WZ���2��G��=��=<�`>C`P<��q<�&�=��5���=';<=�2W�<SP�=��=��=68>���$9>�q>��ܽ@�=�����m���ܫZ<����%�=�q��1�=�C�=䃓=��:�+K��]9=���Ij��_�<�=;6��u�d=��=_W=V9= )�=-��=Z�<�%!<N�=��M�d7x��R=�Df��B�<-�ѺH��J�=��7<�y<�H�[Q�<TD=6CH=�oj�Xĉ�1����=@���"K�=��>m�սp�<�^;n��<�V�=�̚�����V���㢼H'��:v=�w��bD����< ��=�[ཤ�=fI��xs�=��x����=������=�"
=
0�=#�;>�����!=�'<=Q� >��3�Ӟ8���2=�U�<dp�=�~��<I=��R���<O*�=��Ƚ{��=�Ng>U�R(<���=�(;�x�=�0U=T�񽿜�ao���>�i�>��=��=tڮ�j
�=��:=+e8�k�=�=�g?>�8y=s��=;7:<a��]�޽�,��q4=0X
=p��=C�=���*
>ݶ ��C>^H>��8=,>� >���=+�V�qO*���h��~=�Z�=ŧg�<��=�X@�H�	=I5�=��C=�%½�U�=�9�=��K�V#�mꤼ@=�MO��:o=�T=��ƼO�6=��u�"�=��<�7����p�Y�$3��"���t����=n�?�e=J�E��<��=�,�;Qϫ;���<3�=��h<n܉���ֽ"�=&�9=��;p��=TJ�<��V� �꼶IL��ە��!���M<��<����R��[�"����c�հ��gD��@{<f�|!	>�"�=waV<&�1��ʫ<�=�;G2�=���=���=�?Ͻ�޼�N6�C�=\;��N��="8ٽ�$F=�t�� �=��a=|il��"�=O?x=��=��P<�̈=�/ɽ���<���=���CJ=��<?��=�1<']�=:�Ķe=�f�=	5���1۽F�:w�˼w	�=
��<�y����]��ǂ�?�n=�h��=��E=I<-Z�Ѳ�l?�CTh=����w��"d���Ya<\���)=0�N=�jνk"�=چ��SȢ=u��h�j��7"����;dS����-9A��:ooK�E7<��<Py�<��;��� >���<ϸ�󇸽��R=��}�=̓�=ul�<D���=�	u���T�- 8�h��="������<NI<�z <�6��Fk�bA�9�0�|���A$k=L��=���=g]��z�=(�2=�7�?i���ۻב=�<��=�ν=�r���,��C�ֺ���D_�<��������>+rD�Î�<�u��&Ի�����=?��=�=G��=h7=�m��������;ll��m;�%�F=;��<�/Z�?��=B��;w�v���"<0�`=4��=n�<��k���<4�$=Hm%��~�<kG=���{ea<�i��U��:�B�=8b��I�%<5��=Ɣ<=L�<;���/�<<���:����l<V�"��4=#�j��*Q�tmƻ2y���L|��-=�G=S#(��w'�Mt	="�@��=\<�ټ�\�븹<���<N<��)����d�=�����<�:�:h�T=G��:�h�=7�=
kQ��
�=�J�=��]�T��=�;�=�zv<�����=�~=VY=7ń���>�.n�X#<&<�3u��0�'�/�=�xe�A�o<�V���O�:e��Q�=
~�=��<^cW=L=fs=�iؼ�bL�rE𽷎�V/.=2q�=�ݼpo�<����;�<_��=5�t6J�I�A���"�~웻T�����;}�q�H��=�6T=w��<-��<c]=wdZ=e&�<@PL��QĽD^=�?���I�<V��<����+�=��h�Z
��f�1L=�H��Q:�Z���H�=ػ�<q�m��ѧ�?��=��8��=|j8=�Ӽb�I=#�Ƚ#�н��m���U��O��<]M=^*@�{=s��=σ��\"�WS�<W�׆�a=H�(>�9<w�p��ޔ�'�Z�>���`�;1T�<3��<���=��=�E���y�r[��v��.�<��<�g���&�g�*>,"(�kyo:��=���DF;�*�<��=��=�h�=,MJ���=��<֥ɼ���=��u�� ��`���k��2�;�+*��:��U��lm=R$=�9=#i��v��*�x=����=h6=!&y<-��<u�f���C=�x�=��	�~�x=��(�F���b�=1<�I�0�h�5��<ļ��M=��̣;�8=��ڽ��,=Nt<��:��-�<��==9b6�+o�qs=[M�<�<0=[�	�C�=�eF�6��=�&h</�<f�I:�ڼ�e,=@"�����<]��,ɽ{�=���<��k������0�<��<!��Q����=��=AÜ��3�<:!=o2�<�<:���`=I̟<�ԑ�B��<�v�=�}=;�,�=�Vu<���;f[���`���9�3�E���=Q[�<g�<�f�=��Ƽ,�=k��$��E\=��=��S<���6q��_Vּ��=Јj=�p��M���kk�<�	��l�c��7i��|=�T��wkd����=Oa{;�l������)~��?�<?ߝ=~���1��<�ч�\/;��T���ݐ���q���ǻ#�= G��;�V]<=��=3$�<�]>Hm���&|=<�)=�N;�5��<e%<��=y ��5=�X�=��Q=?b(�[r/�KV)�H��=���<��K�Us#=�F=��V<(|=�+�����;�f4�W�<��R=p> >�7���<���=5l�=����j#=e)ν��<�&7<��;��9���<��
�`Pݻv�;��~<�w <t�׼�;�&����=A��<��<��=�8� N��r���C~� ��=�b�<���G��=�Ű<(�<'�X�F(�=�F�����<dV���[>=�����,<�Ӟ;CO�=�+�<Q�%�ەw=/F��i^=.�f<=���;YCȼ�U�|��R��9	Ƚ=i� =ioѼ6���{4_���:�c�ԇ=�H�<a?4�_�#>�8�<�8=8J���޽�#l<ݗ��M���bH��-�=�Ga���~<TW�=䎇=3p����:z��:h���ԙ=7�����=�ҟ���<v�;�==��;�l����^�="��w =.`��a��3.>k��7���c*=������=Z�e=���< �＆�= ��<����A=4<}:n��ڸ=e�;=(��==�5��2=���=�m�=9X���j=�(�=z�q�%P�=v�"<D��<z��=Z}H=z6�=ghx=38�<��_�e�r=1@��&;ND�;����G���A�.�R�Ȇ�<��Z=�k;`35=�L=��z<���;� �=,��;��=8Ri=��n��E=�6=��;�O&����<*/��["����{�<�o=��=b5�=��;��=���<��o<�E׼%�<�w�=C�=x��=�;S�[�'��=����N^(�`�<�(_=v�����=�5�<��)>�(;=nG����=���<(�=�I8=�샼b�>��<�㎽���<-i�:�ռ*%7�h����(��ὗ`�_`�>ӎ=�|�n(#�����U8�=i���-��<6���f�g�_�B;$=i��<�'��gq�P~��q<Խ�p��>D=�O���}�=5�)� �E���'��jI>aK�=�i�� 3�<��3=.h��{�Z�6(�=eq >�)<�B�P���ؒ�;`7��Le��`;�襼DpX�E��<\|�<Mۭ=&���l*�_G3��=�+�>F�ۼ爌<y�>U�<ȧ;��\<=UL��(j��[����u�>�� �+9���s��A`����=�>鏆��h�y���=�L��.{y=7�Ҽ��= ���y��=&�=�z��a=������&=�Z=$Ik=^�t��8�<����Ҵ=�=�)���c�<x��7q=<�=�Hn�3��=8��=�� >4��=fK�=���=�4ݽ����q�_�������<?�=�܇�*0���h�=��> #�=��_=�1��RD�U�z�ze޽�n>���kG=<�Ǣ<�7d�<2�l���Ƚ��7�9u��k������<gIz=@������<bD�=�_5<���H��������=f��f�;�A�=����нH�Z�՘�K�,=��%w�<g޽�7W<�=%=G�@=�&�<#D>�h��b�=��7=X��<���ҟW=C�����<��~<��V=l�9<ĸ�=��,� �=�N�;ڿ�����A	��3?<�7��rD�.��<L]/;<��7��=iӐ=q�Ž2M= <V=}X� �<c'��E��9<�=#!m�5�$���	���Y<�gK��b0�6���l!	�ѷ<<�<���<��R=�6=�/=�΢<��t')�G���2���8�V<��:卩=pI4��4a���=y�>��$=6���i=�:|�	Y�<u/Z����<L�A��ĩ�}�g�:܋�ɱ<�$�&竽�o�<�&�=Nl=����%�<�~�<i�=�?�3��= �,=��'�,�=�q���b6=Gr�;��)��0�<6f�=C$�=�1�;a<�Z�=,ý�z=A����ʻ�[<��r�d#�<�ܻgx��e��A0<���\��=�5/�d�|��a�=k���P2�4�d=,�=��|=�����=�C�<Ǡ�;a�ý�9��J��",b=M<�]�A;��������v��=�E�=U���0>=�*�=�0�=4�L;���<��	<�˽U͞��~�<�́<�Q���u��L=��`=a����%��Ur=��~:$�=��8��,�<���2�p<f`�=l�R��ȇ<�����E�R�i=�i���f��>�PS�����.;�l���U=�,�<Z!�Q�Ž�>�=�eļa����f=����*?=��t=o���Ƙ�|�����нj'=���,�����=��4�|>C��;Ѳ'>�
�<�/5;��^��Y���_w�gc�=��+;���=[����Ò=+D;�ȷ�g��[<_a�<�T-=�g|=u伋Y὎(����=��<QʽJ\4��.�=�%r����92>ȧ;�r��� ����=�%{<t9�N:j=C��?˽�< ��-���F:��}�=|�ٽꕥ�]qX�K�V��J��DC=������뼆EB���]��5%=����`�6=�0�=5w�:� �&��<�\>=^�����Fj�����=�=ͼ!����RG�P��<�r,=�F=IJ<�"�<Y�<�L=e`�1�=�60<�;=�
���˼��w<:�1;�*=3����<t�=�`����'s�:��ڽ��>��O=�U?�|R��R<2�=�l=o ������7X��s��hj�uǰ=b�6�W#�<�#�=.�&>��9�����_ɼ:HS=�D�舣:13�=[V,<-=�]�����f�������
i#=�K=���;��7=#|>0�6�n��<-R/=R��=��+�#α��pJ=� <���<ٰ=�p��3Z��Y*�r�ٻ�8���7弌1> R������E<�e�=j�\�u���5!����>Q��=��6<Mn�=�=z��=8s����;�p;�+�<��>x�;n���=*=�k;��\��af=���r����=��Z=��;h��*cu=�,V=	� ��=�<G���W�<���|=nL~�ު��U�0�X�;d^=��^�����ф�=.��J�=&k)�F�O�m�I=�"�
���z������=�1=W�S�Y麢���]b�=���D�=�E�<Q�<��I���<�D�<dZP=���ې=�F����n���Y;�U<$�@�W�	=,�l���n��7v�섎��=���Wq�<�8&;��H�+�m��Z�<���[�.����=�lt==�ܽ��� ��v��=���<ε�=$r=��=��1��4�<R��=Tho=J�$=t�=���<N)P��f�=l_=苎=14b>�Y�<z`">or��ʍS=.�<��d�N�ͻ�<a>	v=Vuּ��C>�ռ�k>
��9-�w��8	�!<�k��{XJ>�4g=��"=�č�L�N=3�h<tᎽ�la<�����a>/e�=7Mj����=�Qݼ�'�;�5��RV_�h?�=�*�={�=*�5��˒=����Z!>��Y>ή����=��)����=n��<V2�=�˼�ē=�F�=�Ik����=��`�ǳ�<%�����=:����e=�R�:�;�<6��=��s=F�<�����SF�\
o=,�g8�;���<�j/9ܸ�=և+=;^;��6=@zB�W/Ľ�?���R=���(�����;-�_=s�`&�<���=U���¼�;�ji<,��f¼Y�>�x,���@�0�=k�4=HW����L�< �����<�%�$��<Q����<��p=�����ݐ'=gμ�o��y�<�r�<!�_<u/𽛭�<�㖼��(=��<�҅=ĐD���;�z�=YV,����=�J�<w�<�ml�>�Q=���<z�<��>'p=T3�</U�=�Z��)����=OԽj�=�3�<�t�=��ʼ`�=�_��=�"�<�=��;���"�M�>_�;��[��_��Ǽ�FY����=ER���x@����="�K<�Ũ���%=����椽��׽E���:/������=o{���q=�X=Q�\���e=�&��{Z
>��*=�e��g��<Ym�q����T=|T���YѽJ�;C�o=='���L�=]@v�ə=*=�'m=��<�Vj=�4=פ�=���<{�[��T�=p+y=�ѩ9��=p흼,k)<����P;F�Y���=Tx��?�8�
B=o�7��8*<�5���c�{.<0��=cNv=�T;G�-=��<e_=�%��׀ü~�׽
n��j�Z=L���g��3`�<�tT�=�^%��=K;_dX��;=����;��E�=��\��>[ջ�xQ��?n��s<���<��\��#�=���:I�W<�yϼ�>>�=V��RE=!x�<,��=	z=�)Y�J}Z��m=�z��u½KK)=#j�<H�<g|l=6=y�=GɻEI���hؽ��&=���"���K=î���k>�9�<j�@��=n?�=_�=��Y=,�=4�<�Y�;��O>pP��௽�Q`�����ýH�->p� �I�r�S鄽���O�Ӽ_j=��<�B�++G>&��Gqu=���<�y;3'�=����%���6�=p�<P����+����\=�D�<���=�]��{��=&�<+=�҄j<�_����=��6>;_*�t�N��>��<��!>Zz��2�<B����:	4}�;��>jv�=AL�=R�ʼ�y=}~<�[��[�h=��S=��O>�%�=w���6�=)�ڼ�+9;��6=�D���c=���=Se�;2�뼚�>"�J�È>3�>u��i٩<@h�Ԑ�/��<�ޜ<�>c ˼p�>��j��8=:�Z��Y�
�A�k��<���=��5��ٯ<�#!>�Ԅ=ކ���=ڼ�w�;��=R�6>������3>9v�>��ܼVH$��>�o۽�>���=���Kz �|�e=W�ؼ�.>>���=Z8�=ІǽO�=��(<ſ`���;=���=���>�TL>�@A�팰=P����O���=jܞ<HC/=�{��:��=�N=<�=)[��@q�>�~�><a�=���=28
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
value�B�@*� ����wb<M�;��9<S�;?�<0v�<����k90<�0F<�ߟ�9��%�<�#�x3,;����m<�;�:�;]�<3���л�sX�ҏ*�m��<gRy�d��U��;&eƻc�<&������<Cn;N(�T�&<�>+�~��;"��;Xe.<t�<ٝX;��b<H����<�!�<)�"��=��e	�;Oaȼ�a%<�d.<����r/�;ǩ�;S�l��d�<0nc��l�vD<��O��:; ���H�La��2;
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
OStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizerف
6StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1Const*&
_output_shapes
:@@*
dtype0*��
value��B��@@*��iټx�<4������<�P����$=�����-��Tw'��=�`+��n�;1� 1���;��-="���Z[���>�N�|��C���K����S�SqF=D�;5|<䶟=+�@=�[�=7���=���4�="�B��P=ٔ����Ti
�M�=ecI<�æ=t��U�n��=�=��0��=��lȉ���h�:��,e��[JQ��w�<�n�;�	P�� h���ʽ�_:=�C=\	%�T�Y�ɸ��/M=Y^=�<=���=���<Zls=ݷ��1a[��=�+ҼY��%5�=�.��
�=�<b=�Uٻ��=�C=��	�]��<�o�<�W8�p��"�=�%�7𹽮i'=0!���l��Gv=�n=��P>�3�`���u�>����@Y�L���v�<��»�s:���=����˥�l��<��<iM,=�(����<8�6���u�M;ôa<�����SA�rI�<�'�=�^^<ٯC�x�=�&=�/"�{b�=��|�#c3="�� -=������6~�=���5q��)��T�<:�B���4=��HZv�ƫ=��<�P<]��=���<кн=���vܑ��L�<}s=`����1-��8��t��=EQ>9�;�(�� "�w�(�ӭ�H�<�{`<��I�1��s�=>#G��p2H��#�=B<�Q�����=�(B�q�$�ҞY=i�7�S͖��W�gD<#�=���<C귽3t�=ں��}��<� ���=��5=s�`���g��$˽�z>(%=�� ���h=�%:��O���=�u�;XVG=r�D�ի=2�b���@�2�*�{�e=Z�>1�(=����	�A;D��ܬr;�Y<��R�e�F��=�>2��l >��� il<�M=:P�<Ў2<�ý��j<0���ض�<So�<'�P�y�����A�<2m�<�ue���<B�ͽM�r�֠
=x" =��_ $�įa�rd=��Լ�)`=�Q��_=ؽ�=1f=�L�=�X�s��=ː�=d"����C$�p�L��cZ�D������;�x�N86��a=%�ý�P���ʇ������;:�SJ�<�C�Y��i�����t���5�<�[z����)���>	f�=�a����f��A���T��>��T���fh�P�ὐ�*��E�� �=e�8��; �>p�=-Go=9�7��Vu>�n�9�.���3��''=t�j���!�]�"�d�=3�=5�v��d>�����<ZHν�H�=��a:������8�����t>�R�݋`�w4�=�x�I�	=��K>�#���<��<Z�=��<��u��b�����=���=��=���b�5�=g"�<�*z�C���O�q3����=n|ռ!+�d��=��<�Q=1�d({�L�f�M�7=-�>�T�=��<N��=����X��J���^;��?��=�.�=X��<�I=�$,�1%P=lO�� c�u���QG�����м�"<��0��E<u�=PU=���=���<V�H;ʰJ��ܙ�d;�=_�f=Y慽#�������r=Qt�.S�L?k�mԍ<ξ>���I=��<;� �=��}��~���K � |� J&=ӹ��2�(�<[q�=Ǳ�x�~=U��=ƅ�����;�5�=���<��6�	�<����|��<&�>@=�=���<g��8���C=>��ļ\l���5�1�2���<=��B=�;\={KG<p`��a���.�E=�Q�p]̽Â��!v��=�e���zC<��=g�>C_=����+�;e����H@�q��(�� ǂ��Fo<E��<���%��=��E<�=֏�����=�.��������=SE���={2��, ��#;���WH��(E<����O=���_�t���;Ǆ���<�|�;5�=��h�gg
;�`��.䌼�!=I�׼؍u��e<���`_�ڻt�sd4�w�����ڼ�߼􆀽:�$=f ��� �<�[g��H�s�*G�=nGý�����	����=�*��=�.�L#Q�(��Ǭ�;��<��=��T����+�2>n���h������B;(�=�.G�������8���x9�g��5��]��(�u��<��=���1/t=��=�h�=Û�=P4W=�ӕ��~�<��=�+;	�/=y�d�X߼���<<�W=��۽��i<²	>��U�ն��c����H�*k�=m�=�K�<���	�6�qG=_	�t ���1��=��(=`�=�cw����'_d=�)<�e5��oG==a���{l� �ּ��}����;b�=�H���*���Y���VA���=��&���'�X��<L��<F0��1���=;���)o�.���Q�=�چ��r<c�{�-�=��{�����ߺ..�;�|�=�����=|�->vn���N��~�=�t<��<?�>�$�M=s��^��>�뽻jM���1=֎<?8E��[��}:��kp�]��m���o4=6�i=�J�=A�u�&#*�������=��u=�8L���s����<歇< 9�i�=���<��\<�b;���<�*�<p<�`��Ȩ=�,%���c =w,/��L\�P��=�盼�Q��8$�<s�Ǻ �����<�1��x/=z�L=�i���Tr=�����y�=�#��)~��:;�C=	g���~�}#�񵓽��u�J%�=.hG��=�Т=�=����"����Q=C�!�1���=9p������KE=#ǉ<�d=�[�<��<U���L����O=^��Byx=�G�6����ѽ���=������+��k<�=�5�<8�+==�x:�8
=�A��� �=��u�m'3����Q!���{��F�.<����<4[⼴r��ќ�<Y����1��rJ�_�<�{<��)=݈����<�N>=b�<�8�9���Ώ*��%�<Ot�<���ŻW+���� �m���CL�&O�<��<�Z\��=мNF=��$�K=\�*����<�ó;�1���r��t72=3A=�%�<ߤS=��<대�d=z�V<�C�<$ `��[{<�Ѡ=�9�=g�O��%�<�}=;�f����>�z�=��9��m��'��KL��+Ľ�
>��&�L��1��<�%�=|�ч�=�C=Q��G�`<9�8=�Q(�K�߽��K=OV]��#���kF;��<>�.=��t�BѺi`=S�=r<�<�|=��$�Jö�D/1=�I�[v=�w���"���=���:� ���I;�(�\M�<� =Lg���>��=L����2=x&]�E��~r�����RR�W=���L<WP�=9Sڽ��R������k�<ݿ;����[iļ]�C�| ���X�l���v@�<��,0���P�,|!>�">�v=,�o;wK'���/������NO��� ���ѽ������!>�G|�,(�Z��=����6�Y >�փ�����E�󼈉�=+a=^z&�����o>�=��T��bH>��<�a���I�1����%��<����;;�W�l>F��= ���^�]=�l�,=����=x�"=�:;��J=vE�=D18�:��e�=�C� ʱ=�n�<�9!�+�<���؀���I(���W�/[��8�,ل���)�"�b��W<0Ѕ���E<<Pi=EG	�:��<�Z�z/M=D�6=��=�=�7=�gy�7z����b�ȼ�?=�Ou�����l2<������=��T<�̽�&��%5��b]��/���y<T.2���<n#=R(¼i���[�\���uD&=�%��@ ��������=���<6,:<��	=H�Q=�u=�<~� ���=���=�}<��
S=D&d<�*�=(�9��JQ=I�K=�K8�6��=,<��I��<ܒ�=��<�h����=�6\;�j�<��Q=<z)m�8ܻ�4�=��:c��v���2����<��R;������=�n����<��<��9�T�2<�J�����7�=:M�X�}��;�{G<����©�<Hf"��oѽӋ�Wu&�e�,=�gĽ�9s=��2�hd|=�W=BN��#��=܋�0�<�{�=�>��u� =�8�=�YT���=q+<y�罺Ѯ<��0�C�h��6*=����Q���R3�m}R�b��=�_��3FB=]�o��xt=���5<����t�%Μ=�)���+{;k�%=i��=)���!@�=`���G�=���<r��a5�����薰=�lE�i�����=�Hʽ�x�R��E0<��۾�<�|=�!t=�Or=��'��}�d܅=�������<�W@<��;���K�=i֭�w,�=j��"��=�I�����P=`�Q<w��=�E?=�ǽon=��<K����ꎽ'L�;�13����g�I=}&`=Չp���/=C�`�2�/=�1�:�ݼ��<'(D���=ң��_(��/�=i�q<���:fC�͇�C.����=�s�� �U�G��;�sֽ�㊽�� �ƽ ��\<�-�T�H�V�)=�B����	�Y�-<[��#�=/�>=/�����%E�=�Y�<�h�<쌼�!C�]-�<��Ԕ� T�M�0�
�{$�=JUr����%����N׽C�)�b�b��a=P�>g��-�,>�U<N��<2B�=���=���=ck�U&�* =�h�;��<'W�`��=���<ш�;�ʤ��hQ�>A=��/� �T� ڼ;'�=��=�`2>r|w=B=�u��Ĕ�d=R<<R�)���L}���Ts<h#p>S3�[x%�ۖ8�]z ���:=�����1��g=��?�G%׽�'��������<cH=���;�������=�?����C=�*��"���:�<bvͼZ�ڻ��<��� Y�=�M>�*�����25�;v�#�ڋ��9.,�y�=x$�kR�=�p�<rn�=�SO����=YXA�a����=ʿ<��<��l�x��=ʻx=xT�l��<q��=B��=�X}=4�a����=|��p"==�С<��8�[�%<uc
���f;�&>+��;�H>��ߖ <�GS=Fc�'����=YIK=sj���@=��=��]=<�F��N���T=GK���G��e=N۽�:;<u��=�!��)�<����U X��Ի;B�Z=��{���Ӽ��I=/����%�Gx�=a1/��6ؽ��=I_i�{i=!=O��c�<M����՜���B=\��VO��/$=��?�C�=�Y��BwM�[K&��3Q='�;��*=��<
�D�<w>�]5� &�=�g���Y&�+/�=X
��c(��Y=,�;���W=��f=�»t*�BW�<_��&�0=�W�=s��<<U�=L���=����=3���i�<� @<u�=�ڙ=��=����\<�}=v� ��=#��<{�-��C�;�v�=�)6=�v[�s�=�X=�D���;�=��<���w'=y���b��*�<�����V=�^c��WW<�U/=H��<(���,y�����E&=�0J�=��]���=Qm���J��@�EZ/=~�?���=r~�=���i�`<��S<����-A��;��=�>�<���<Z��<M��7�=V�%=���=��!��F�;�^�=_�/��r�=OY�=��<ڝ=�B�=ik����=qx>�4�=�E�=;�=�<��t�B<��r=L�?�03>,��<��=�p=[(=u�>�d�ŷ>�9>��-=)�>�>��=�
^���Ҽ՛�=@T�<2{L=���<�~9��B�+V��7F<���f*����|�s������<<�l<ɪr���=��#=<t.=����0gG�/��=hF=Ǖ�e@���=�!���L���=�*�<�5<w�=��=k�(���S���,�Pv�����=˷�<�6{�d�<��;�H��8�<�Q~���<��烕�剄�����b�=�F=��;�}	<�_����<n�����=~k�d��<�=��=~�f=cW�1r�<!��=1�;����z��<�㹼�n޽}�A=xq�<�a����=�=��/��aM�<d����T��Ay"=ë�=,f�=Q㰼����V��(TD=��\*Z��=�<�нx�e����='��?����ȼ��F<h='��C�=-]��3�<6�=���=9Y��}��=K$�=��Y<��A���x��������Ӊ	=E�"=�&���y=�qG���;�Le��Kۺ#�=���:�e�=z��DP=W&h��=;fu<�<�h'_<МS�[D=�a<e▼���<�����H$>N����/	�i��<��<�2�{���=�R�==)�<c6=d��<�>$=^�=��J="�@<#���*=u�6=5� =��&���=|� =Uq="�^<f�(��4�����J<V=�8��Icj�����F}=~Lݺ\<��3 >>����r<���9���Ⓞ�>	S=F�=�{2��x�<��/<��K=v�=@��=�!d="CX=V����/$<� ���X<�=�?��s;F;sz0����<�N<|��=��T�@y�h�F��@�=X�;�F5;NMʻt;�z<��=v��:z�B���t<��:����Dpv�@'�<y�o<��y<��X�
>�4=�/�=�kM= �;=��T=����fл �b;v@��]�0=ǽ,=��;?a>g/�<���,�=i� >������
=�>�#>�����)�=��l>G�[��L=���<1j�=�$ͼ�?�=��O>�y�\|r��=�;=�2�=2�=Q俽E���,����.�q�=0&=���������"�"��=��=*��9� �������A:��U>j[�=1޽�L�=Űû��=���=� R=��sO�<M	�:r<[��<��߼!���<�=Ш���=�"<=j�=�_*���<��=�W��x�=�	.<����M;f;ʅ�;��b	�x��<`�4��+����y����I�h�I=J����=:?N=��\<�������N�=_R���X-=m��=���<Z2>w����U��>�ڽ(1���ۑ��=�}���@f��l�<s�4�]|�G��=���-�_��nλ�+���>:��F�<*�=`cl=���<@J(���=:�|Z=j���m����=�Yi;���J�t��;Ј�������FG#�<�U=�bD<�b�T� =����8	+=]�>�o��z����\���9po��J�A�:~�=M�½��7�ܾo=��ܽ'2= J�=VV�<�I\<p_ =u��΀?�^J
>9t�9�Ľ�G�U*<�a��(���oY߼�T=�^J=�4��mt���-'=�!=�
�=>a<�6���˥�(ϰ�>��ko��?�=nX=M��p@��*�=ռ��&='��=��p=��+���Y�>�/��GG=��<^�=~�����\�<b/ȼ~�=���р*�H+!=�gҼ�i��o,���:z<�6ҽx7׽��B�˯=�v��]w=�T�d���T�kgػ�H<o|�>�=�ss�1�ڻ�(=�	���@�H{�TK��4��#� >��2����=�2���6�G�b53=�����Q=ˎO�����>�(#="㔾||�;f-�=^W�=lՄ���4<P���fo=��F���b����/���@�<�F$<#�=�U���⭼�l�;b��������=�s={*���G�fʔ=�H=�t�Yge<:�~�'=��)e=��;����h��dc�	�=�`��kV9��W=�nq=����]�<��<�\%=R�<^��_V;�ƼʼR=�T��UA<5�0���=�ȣ����<����({�l9�<�|����g�#���1���;r�<@&�=��Q=����^j��s=7Ľ�UŽSM=-�F�唉��#V�v���B=[� �{�=�Ue�O>�M���ei�j�<'IN=���=���;ł�:��˼�޼��X�����<�M�<��˽c�k?�<�%B=`i<�7Q�U@�=~pS=����3]���ڼ�aT=��W��I�<���{�=<�3=��=>��=�ML=L�=e�=)��;P5;��<2�1�V=��ż�FI�c�<�"5�p+h�Y�=]�=�����;S>�p=�wR��^V"<�A�=�j=�q�����x����<��=������<�ʫ����By��]�<�1��}�F�F>ig=�_��+���;�i���i�vQ�<�ٽ���E�=��=�)��_�<���=��=j���8�~�GZ=���꺼/�
�)�/�N�>���=m�o�^f"��>C�"��;�~��f���zNg��C���:��;k�L<���yaŽ�f�c<y�H�>i�z'�-�=9�߻���<Rj��6X(=���=X,�<���<)2�<��j�ϟi<
��<�͒<�L<+���M�:�M�;i�ν�u9�9�9;k�I=>�;��=��=X(����ؤ��Q����<��=)��<x�/<W\�;87�=;6y���,='���L��&�=S��b�=�&S=|	=#h�;��<=�+�=P�=w"z=m!x<M�]=T���K9ټ�i������߹#͓<+�%;����F��"8=^��<����:�?���k���F=�����iYE<��=%;�<3
��C"�<���V��<�B�<O=��\�>6���39�=������={c�.x��J�n�M?�<�>�=�
����>��<lkO=q)��oӼ.P��J�սֿ�������8T�=��=!��<n�\=V~)�`�>�"����g&C�|н��=���:5�<R�Q�-w��Ю�� s=|c�=��z=�$�;��R��t<�Y;P������Q���~�=��<�Ϸ����D=l<�=����Z<.F�<�j=�Ž@%�]܂=�!\�0fӼ?�d�s���"ż#;�;�(#=�o-�r0�*���K�wN�=���3�6G2�4RD=%����2]���<���;v�<?=�����N��=��h�Q�мJq_���P���__�}���������g{�������< [���Ľ������<�<�rr�;e=8+�<��<�Z�=�x���<=6�<��	�O+n<9�J; ��<�E輶����~=ߑ��s��l�m=�?<���;��=�޼���^½�%��ۼSw=�;�E�=�V�;�g=�c��+2�;�<1μ���=˗�=[�W���t;c��=1'Ӽ/��<�ڡ��߽�SY���=�8h���ᘽh�<9$Q�`�E=�)�=k����<9��;�����<0<<qn<G�=@Bۻ���9r�=M�ͽ6xS>����<L<=��Ԣ5��nh=	i�6]��n4��c���%9���?�u�;2��.�<=�-=��<SnȻO�����;o>��*=���;}��b�<n���A�=`���D�����o��=ʲY����<��F=}1}�y�
=��=����r�����;(q�>���� C='�K�/�v�μ� +>W��E�<4=]c��o�<�v�;ɒ����<#��;E��;M����t�=�9�<��=;ʽauϽ@����>��t���ƻ�ט���U�r�:ݏ��X����=�3h�~���]�=�=QQ�#k���a^=ڙؼW3�=^�ʽ��.��>�4=V���.f�=���H���
�==�����2��B�����= ��k0�<?g�U���+�=$y�=�C�W�w�*�<�R��g8���/:E����4�u،��O=�a��>��I�f1=��<T=�<<�[�����3=�+��������g=��5=�j�=>r��>c:�Kb�� >	p9�< <4����)�$��Pƽ�
l��!�<�r��X�/j=.����:H����<�_�<Um�<����(���<�䙷=D�'�#F���=��=6����=v�����=���1>�>�H��n�=|�Z�{�>�8�uh��^�?8�멽��6=-����b=�;��s��F�1����<|�ֽ�>�ā;G��<"�<EY����Ͻ�	C�繢="'�����ZI=�ۋ<h�;}��V!�ި\�ux�=s4H�y�
��u���9)������m�<�ݟ���x=2��ֽ��=�v�q'��Wk�Fht=�S=tf=�=�û�>h=`������>��4S��!�Ї= �<oa>S���Gr2>�#��=�ϼ�K<vb)=�mE<&��@L�pB���ޢƽ�t��>ݽ�;���*�=��R�N���x=W�ü�}>Ũ8;��k�l�+�~YE��q"=R�>�Ϩ=`�=�A%�<����R۽�l<��>b�����=�J���H��Ƕ=dS$�E��%�|��t9�U�5�OX!�6�4�qYT��d�=>u�<<=�k<�Y_����=�t��% =�覽 �<<$��-\k���=�H�����<i�%=��r=��r��%�٦�����<:��:���<����.޼����)�=�F�����Xٽ�I�\v�=��s=|���8&�<�!��2�=t��=FJ����<:���=>�=>����=ΐ@=^�y�-}弍S����I��p>oÚ=g�=B<���U��<e<�;��5<���;� _�ۚC�9_�dr=�����t�d���gy=��I=��X=QMP����=<�ؽ7q��=�O<F5�v��=���<�+5���i���;�_��T��;ݶ�x0=Cb�|h;��%�3�<'#R��엻D��<M�e�܏��i���-���?�뢯��F �:�:�7�=Y6"��a��=�M;�M�;�Z�=�Y��c~ϼ�y=!��=���s�P<H&,<�^�=)�ƽct�=r�=?��<"M��5�'=�:��6xJ=��=�ϲ��3����?=;�M<�c<��2����;�\�|~.�a��� ��\=sJ:=fJ%>뻽(;�&�;E�<#����=p�g�w��:p��[�)��F;mz7���;�I�>6�<�� ���R>2� =��=���=V��>P䀽�p޽:������<=�4�=zP�=F\B;?�e����ޓ�=#oE>ԣ%�}NŽ!^��q�w=1>��>�㸽h��<vY��S��=��9=�!�<a���Y�+�g6H�M��>�����xw�) �<&�D��� �(>�<(4Ľ['@>�m���#�;�༭�-�%<Ƚd��=�w�%�u�혻L��=}�;���_;�'	��b��y�=��$�H�fQp�� ��`�f�̡�5����l/��_G=�|=�M� �X=D��;	��qyK���B9fQ^�>T,=�W�=�8���<���=\4�;3==`�ý�ֽӟ��BH�=�0�;-�3��ݒ=������=�Q5��Zȼ�=�$�2� <���= ���qC�i�<���om�<�V�;�X��#H�<t[
>���е=
��=��۽	�&���������&&�<И�<7�#��g�����O罛L��jW�7��<	9��\�Ƚ���u��<洁�)k���߽B>A��[>8/=�P;8k
=ﳈ�z��i������,����<�H�<)͋�'��=9���~>!X�=�9����ʣZ=ٕ��;�;��ڼ�Ζ=W��\T>��N=`D0>* =������>q&�OƼYB��;�<�M�N����
b<����M>꤉=�07���=�I]=��D>&bX;��==e���-=��S��Hڽ3��=�ݱ�4�=n�4=�>������<��?>�q��r�7�ж.=A����~:����=3){=�c>�kݼ����~>�m�=�@�=Y�>�=[+�=r4�� �`��v�=5�)=�����	F�yJּV�=?�=V�0=���@y�=1�ڽ:�p=�P�<S��"�
�3�:�F/����=�Ij:N[�B�<u�.=�G'>�B����L<Q&(�e?�<C����-�=vV㼵Ӑ=��;��=߀	�J�A=}q'���5��΄<����������=�<ۣ�mWּ#�=�5B=�Ȋ�1�=�x���pp��p�=A�>���_�K=۽�����ռ�#�=�a=|0�	)0=��Y<�M^��7�<(�B=V3!=7�ܽa�<!Y=v��=��ٻ�P=QU��%�����= a����<��!�;�_�h��?>�o���{�_3�=�;=�^�=?�=�\U=�#����=Y����U�<�U�(�=:M=�Ǣ��k=F����Q��u��<Ỻ��;ˍ��t���̽�gʺ#�;��>n�u<�S=Y2=����#h��[z<��=9G�=��8;����+!k=m�ǼV�\O�<�E���.<PUn�����$=�-�����?~�����<b�(���Q��y����oQ���k^<��=u�X�g��Y�<�����֯:JLT�-i����<���=�4�<;~�ÿ�=X�<�p=K����^�\��T1"<n�9X�@�v�V�:�ʽ���������=9�"��=i�м��������=6�ݽq\�=&�;�HP=���<]1�=��u���6��S+���p=�8,�NFF�rTֺ�,/>k�<͉�e":<
������=�d~�z=M=�I>�D��8
!=F��r�7��	��=/<�<����=��R:�=�O<�^�ZD��KS���;ʌ>���=	Y��9�=�'�<)�=WM�=AQ�h�=�R�=��0=m=�=
շ���M���;�A��=5������&�=Ze�=/�o���)G<;�����=��=r���~m;֑�������q��" >"�+�_̍�%�=#$<�&�����=��Ի��[���<�r���ھ�ӓe�%u�<<Ȥ�ǆ�%��<,
�=I�=�ݯ��z�"��<��=���N=珢<�m��ק׽�8�w���(�=x�����ݼ4�=�p+�A��Z>O�<kn=^�<!d��wO=��˼�����:�~�� �,�Z�=�1D�H.c�����C8���D�t��Z[�<�5C��G�f���DEٻ�9��p�K;2M9==�=T=�=Ȅ����P=0�I�^!��/W=K��:9$<�D<AJa=�0{��%<�m�<��Ի�!}���=�_^��($�����㯽�л�=�輼�i鼣�;��S;�*����<aD<��U��y=BJ����Ǟ��=վs����=����,�=ݖɻx]<��<�V=�z�<��l��<�<���=�LN��W�; q(<��ż��<j=�;��=g/�-�MH��/[��GlE=ǘ�<�M����y�0�ͼ����܀��>�#������Gڼ��=EI�m.�<�0�r��=F�L9�	��u�`<!�a��.�=����0��jg=|<�-,�����F-;�����4�rH�n�*=LJq��m��d_=V%�~���(���������n�=�Bj���/�'>2�=`��<���<0�6=�=�
=ȣH�o�?=1�,�er�<�{W�AvW�%�T�4����<�#��C�%>��<�O�ϻH��<�Z�l��c>A
�Og��`�B�=/#�c�T>'��=�f���x=�	>6!���<�6�� ����|�}�]˔��hX�vx����e�1��=��齌o�=�U=���=�����=�������˓E=R�<�Q�<vlh�R[=�a{>,)>j����=��L��(�F�����g��u��N��<W
�p>!+ü����`hd���'�1߽�z��(����\�TF:�=�g&��sŽ�R(�ˬܼYH��sL���7�;񟽦��z������r$=DU�`��f�7Ob=�	h=��w=�8��	x��V�W��	�w�޽vJ,��D�[�=�/�ʄ=���<��<���=���ڋ�7�Y=�;�;3�<�����l��-'=�;��
�Ț>�MP<�����>����?���f��e��J�&T�� <sB�i��=˃>�ڴk���~��Et�+$b����=ͤI������������&U�Ȇͽ~o=U�꼻l�=�4��G���K��8��N���Q���l=�j�4�h�rnֽC��=�=�=�ŀ8� Q;	�ڽ����}�p���
�w]}�50=��l����Q�>ހ������S1=�xv�q5���ݩ<$�}���O_<�
�<|�i����+�=B>���x��<�T9���;t o;�e���U��cw��YT=Z�;`�#>�@��٩�����xJ=^;��EL���"�*M<��=e+�;O�f�ܶ�h��=ى'=`��;������=�D�Yć���˽����ɒ=(��jؽ�Q<&��=��e=����E��g������!���<�%���ɽ����=�>����>d�=*5g=�P$��r��!c>D�����y�>�F��KJ=,=��꼁Q���]>J�=������q>���O��<�̽���'�<���9����R�d>���<?l<����p��<�h�=�!��d�D��<KoẺ؈=�6�<�׼��=������:6(4���=ONO��e:<�֫;�$��=G��<ʶ8=֘m���ܽ���<@0��R��)���e=�_�=#��_��<���<�6���^�:�ﻱ^�5�	�,F5<��=��+��<b=��D4�W��>�л��k<!�A<E*�=�G�=��(��m�:j�ۻ��3�Z����;E�p\<�=���~�x!�;TY��]�<I\��"=�=Ǟ�<&�O�:;���ޖ:���=�2��T��B�A=����WP;/�<���%Z��J��=_뷻�Q��-7��!��8����8�=tᓼR*P�`v���P<o\>�UļC˽�g��[c�Q-=M�����=�j�=_M<u�S�]��6&=�M����=�"ݽ���:L<S�=�<av���=�0�<��<���s����=�D���9�{��u�=Q�P��g8<|���	9c�=*�8���)�"½aN$=���|4��8Z��6���G=�H���M~���>?E��0��h{���np>H�W�<*,�������w�/9U�|�(>�Խ�� ��l�>|�1>Y%��\���7S�|�^����R5��fz�<}�"�p(�����f��>3�/���=���<��E�wǂ�0I_>�����N�!#��ҕ=�`��	D���m�:�>�G{=����%��>��P�����ٜ���s�6^7=E����{=y<�!э>��x<݂������ϴ=\�i�DDa�v�<l|��H��{T=+.Y�`���5>ټYR�=�7⼉�ҽ��ؽN�=ŐB��xҼ���=ۑQ�N���k��=�����<��=T��=��U���ϕ<^�H:Z�i��Jr�N��=KgW��=��2=�r��{j���>Y�M�S������<�ݳ�gJ=�*�������(D=�^��w��[�>In���u�˼.j<B����=�q�@\�گ�=g.Z=X;���Ύ�G�:��}�4GܻxΨ��.컡ߚ�Su-=
�N��落}/<'K�=��<AT=}'G;�'���uƽ| �C^ݼ���<��������Xގ�YWI>)0=b`�����(��̴�����rۼ�ҽ�
=��=B�j��S$>�՟="r�=MU[���)��Q��`c>�!��J=��������r��� �=��<�J �l�;Kv>N��<�޽vlS�uy��vX�1,��/m�[��u^#>��a��²��*���8���V�p'0=�2h;bL= 04<B>�̺�ST
>j@�<�î=N�!>�p=YRT�k`=�v���u>������<�<������d�μ�8n�̩�=��F=��|=s�=4�,���m=
kҼG��=����P�E=�6��,��I�=�5��-a�:�S�<�>���=V��;T�%�a�+�ԇ=}�#���=���Α=���|��=�=�Q»[�?��m�=��ͺ�����2<��p���=	��,63�"�������ACV=�vžaՂ���!����"�����=>�����(�f�=?���g=˄�4#P=������A�l�����=�l�g�G=s2;���#�x4>�"y=�%����N<�<P����"o�R;����������C��h��f�p>�fU��߻��=��ͽ�������>����-1ý\e+<OX;p�����<��E7�=����S9��O0>Cnؽ��=M���B=�(r��1�,��|.=���ԇ`>%V�6s��@>ֽ�����<ykL�у��Α}�<�=��=�\������c=�Ɋ�!^V��P�����<r�=������Y��<)���� ��bR�:a��<��<�O/=J2�;/Vϻv�/��@1=*S���Z=�;����=M�ͻ~MD=��6�<���<�L$=�M�<f[���~�/�<�<�>��E���f��/�=�~�,k<b�w=vzV�)�����m�� @�k��<�t-=��<�L�=kU=`=�5˼�����=�}��6���N�=�=d�H=����ކ	;��T�1�
���<�
��՛ۼ'U���˴���=��=�c;=��=q'��h�(Ԋ�N�=��<��ʻ�����x��W�I����Mw;$[#=����S�\�r,�<-\5��5�S�u��x<V;m��K ��0E=j���ی�Z��=�[=�:	>2u�;Fky=�޿���<F����j��!=�Խ�=O=Z���D�=���b��<3�=��F�<�i�=rW�����<�BZ�V�<U#l�[�Pe�<�]��ж3�TX�<�_��̤�3�<.����@3=w7�;�:f;�C�w��<�=�>�<<���m�=(�<��׽��<����m1� ��<�G�s1=���;Ay��=h�)���=9��4╽t�L�� v=@ڔ;ޕ�;=D{��u�bv=C�����<��G��H����V<��"=������r��!��ٛԼ�v;��'�o����=bqg=?b=�V<��	�<C�ּ�_�=@�<��6��^�=a�i����=
^�g�ͼ��=;�e��8�=?CU�1R���p=�N��O��<ɽ�=�[����=�$�=��%=$��J��SZb��=��>���<:���[4=���=T��=�=��=�ݿ=���b��=K����N=T�=Ή<=v�ۼ�����H���˻��;��==V�=!4�2ڻ�*p=ݔA�w�� ���F���q=�^�=�d�=7�����;�<wYK= ��=bQ=V�c=�꡽1��<Fo�=�+�<0�<D���u���f�6��梽d��=
F'��
���^���V=�==��&=�.ί�=�ǁ=�(��y8�t�=x�]�֥��{�~=���=���ay�=5�����O=eV��;�)=ЦҼl�T<�Dg�� �NQ$����=
w���5����j;Oi׼���H�Y{ܽ�D��)`��VH����;0��:�S����լN��#2����=^�a=J�������sݽ�'�=禼��w=�����&�%o�z�
=��L=gA/;�Ȳ=䤷�ݍ�= �=D�>�IW���M=�9>���cμ܍�=}��=�sA<l�=�V�<�0�S�]Q��H�=J��;,}�������鼯�=��7�)yV=�ߏ<ZNֺV��=M����ß�ŷ"="�=z��J�=<�=��=t4M=�F>�����'=��!��M_��ժ<]�=��8F0�<����5R��E�����O@>�&u��<+ռ5(��᷄=ǰ�;Fe<	���K�?=OEH=�й<֗�:��}<�R0=��^���Rr�F��Mݔ���=�=~�������+>>C<5�=�=W�޼��r�����=���NaM=����'��1S�=�6'>�
.=2���pu=v�$=������=�bn�Ҽ��X{6�˙=�Wz�?��������Z⼏@B=�t����X=�/+�4���x�ڽ�m�
���=���<oT/��B=2ֻd�=��=���<E��<}O�?ɏ<V�j��:j=�6\=��f=��<7>I�<.h�j�k�i�����<>u5<�}�+��<�i���/<��h����=K�|���=�z漟�߼�~=D���H�;۔��33=���=Uc7�D ?;�('<��;r���,�=�<+L��������9�y��<��<j	�;Z%;���;�L=��=�{�=ȓ��0��;�r�=�]Z�rݶ���μu@�� 뼸zq<"%=�L��Ŋ�=�(�=K��=~>�>�<�1���o���N�fĽuY�fӇ=ﭤ=������=���=�)T=��`=ܣ^<��<��<	��:��=�(
���z=J%=UK=�m�=4�)>�X3��/��㳽~Nw��|=ʭ�<=�SO��o<�c��{�=�U�=�7Z=�cG��U�;M��=���:�⪼ȑ���<Y;�=��^�>�{�uߒ<[}�#��=r�n�P�<������Ǽ�3f��Dٽ�׼�����;�ؘ���Ȼg,���ՙ=�����<�ʼ=���۴�'�׽Т@�0��=n�=o�j���=O��=S0>�I�=�t��<Te:g/��E9~=��˽1DI=Y�?=�3l�}HT�W�!<����>��r���=�T-=���?|�=Q.�<@Eo�}e=�s�=�1=����-_=�����,�"�u�G��<uj»���3��<��<�RO�|��ȳ�M�)=$S�<������S���!~B<�$���{8����U	���`;ɀϼnu�Գt�a�
=�W=1Û=���<�7���F�<x����ܽK�Ȼ�창M/��A�<y��܇�<�-�����>��=���<�I�<X;I$� ��Ä{=S�L;1r-�Υ�����h=4e��ϒ�m�p��|��T���=�П��a��N��$^<��<x^#��E�=�R=�\�=U���q�=<�<P,i=?I�<b?���N=��Ľ����ڷV��ٻ�.#��<�=jiN<n#m�Mz�;��=>�-=�*O;6
��<��<�!=|a�A�ļ�a�=��w=Ci�=%�>��F�щ��o�<d�!+�=\ &� ��<�����ɻ6|w���=(�<�ػ�_�<�6�5��=�.:7���m٥=���%ұ<����:�<6�V�]{P�\$��,��<\%1=f/�B�ý��<���<��&�k�4=�А=ߡ�<�g_������t<���<�<4�������h�[1�������=EP
>U�<��<ϵt=�w=?���^=��=��<�����><�L=Gf2��X='�y��J='a�O��=�wg<l�9��`=5h=�ϲ<DQ�<�.?�����Յ�=����_^=��=��=&��R*@��Q�=�.��%��w�l���/���t��<��� |ٽ[F=�i!��_�<e�=e�<5��=������_�������L���bB�=p.s�
0����ܽV8ѼO�}����<�@�<�_��*�PӼ���=��9��n���i�<��<IL�;<!���<�=��T=̅�<� ��n�=��;������p=�S8= ��<oǼ��5����<���<p����,<����A�����k=���=|H�==H'�`e���!>�W���L#<�>鿴�z�!;��һ�^H>��� ��;@�;�*��@=�1<����ɋ�<E=C[˽$!�t&6>��{��N�>2�I�/<��=��=L��:5z��M���4뽒cp<T+���/��`�9:�#�U��.U=�抽���(�g:����q��=�rn=��m��U�/a<�(����>�;ԉ��o�-�"2<����CWy=���)�;b���C�=N�>Bo�<9.=J"e=���hX���=X7�=ӫ���ҋ<����_v�DS�<�1�;�"/���_�@Q�;H��<y��<�h<�v>~�����9=�T��������jg=~�K��H���r0� `�<_?'��( =���=��=����\��9?�!�Ҽ�ߏ�'{)<�v��w���˓=�q<�>	k�;��0���=ʦ\����=���N&�wk<6x����H��R�<҃q�4+!��A���J�<�W�=)��=�<�i��<�����.�V)��oe,�;>��Pv��]>���=�iN<B{�<.]�<ފ=C��5b�<!�K=�j[=k������R��;s�=*��=x瀽���;�v�;�|-:x%�<I>�<�#=`N��
��-�<�ww�ϰ˽�R�S�=�UO�<)i�( �=_�޻��=����L�7>�u���<�o<��<��;6�; ��:/>~�;b�<�&�<�U=p��7H=��L�pqȼ�����d�=�
	=t0����=�D�=���=��F<\�=���=�k����G�ZI���&=�q��ʇ=�,K�S���	=}D��BXZ=��b�^�<�J�2^V�+#�=j�,='>y	���2J=�8�T=�����gz�N����᨜=��$=x��l<���=�w>��_��8�<��q���!���C<����t�;Vڢ�;��=�kl��Ë:�=ʧz;?y�<a�(=W�<��<�<����">����sT=֣=�	=E�?=�M=�O���e!���.=M� =ޫ�<ͽ�=��=\q=No��lIG�<d~=Q�w��=����_z�t|m=����Ѿ=a�+�M�g=�[q<�����ùH򡽇_Ӽ�1�< m�=�����!�	�<�{G<�H�����<���<m�<�8ӧ��f^���b�+������P6y�<L���e�g=�9佼)�=��5��OI=%@�=����ׂ=O������:��r��4���=��O<$�6��� ��6м��1�g�l�8 >��t�z�<Rߞ���g������1>��L�'���.���">F��=�$>�)>�����J�<���=�ZŽ����vvN���0�6��=B���7͞�< �Z<S<�����*>�=���UP<��+>͐�=��=]��=[ ����;H�{=z�?>$�"�,��:�$��=q=�1>���=����Mh�=(�=M�ܽ�ݨ<FAk�f����k��<	���=w����*�ħ<{+ɽDt�=�q=�y<�b�qX��G»��=�F��|o��C�=���K��7��=o���r=���z�=]!M=���=͒+<m�<�>f�=��Q=\1=,�s���=����\s��V(��h����&V-=[m�=�?���_�;�E�=��=��=2Ӓ��#�)��=�Ht<R�1�D�<��;x0	;�3��fC���vZ<�����/<7��:���HM�<To���Jb���=�5����;��ܽ 2�'6@����=k������=��c���:ޠ�<�D�=���<a?���z�=xc���<x<](=r������<ͧ >0��=�͛=�K�6j���I�cN��¼=L`�<�<��='3=�o���W�;�0�<~6�=W����=�́�
 =�����W=��=��z�<�=�'�;gJ����=�N�<~���<��S=>b�����!��oȤ�3M���I=$��<�'�p�p=���<V�:��L�Μh=C��P��/��;�<XZ�k%,=Ρ�=��e<��=�e����4�3�=�uL�j����y�;�d;=�/V�$Hx��@=�H\=p�<��2������� >6�ʓ,�6Y��ge�<��q��^��1�<����j�<r���=V�<����뵼����%=��M:ܗռZ���m��Hr�<Ž�<o��<f3�m��<�����P7��\N:��ȻI�<�\�p�=�7��V	=��=�ռÀ��[�q=��?=��<�EƼ���=����x~<�>)�	;-k=s��2(��!��2�Y#;J�<"�=�=��v:���s搽݃�:+Y�<��q��K=@)�;2i=&j=t��;Q���@0��.=G����I�ܹ&��f?�a�+=��z=�����%�ܼۡ��S=!��q	��^�%*���T��D�<A�'�u=��:ؽ�<��=�J<<�ҽy��xo;�i��<�J3=����rf���%=l��=O(�<\���h����E�����Q;м�#
�� =���<V��=��E=�N�=�ŷ=@$=�_μT��n��<�0��٩t=+���Fb<�:=0u�;�O>��P���>L�;���<�:��o��h=��B=g�H��C�"�~�uW���ט���)=��<�}��{�:�i�:��(�<Oi���`�=Nu <t~�<mx=��=$\�;�<q��/o�h��<��<D�V;�(��41�;�r��jڼ�"P��|>�&�8��#G=z����=�(�x�=S����><R��<�5<�R\��N>��3=ۋ�<�TF<�B�����<pk==-Bj=��?�����s����=�Z�<�F=�5S<<
�;�*��/r����<�`O=!>��ѽ���RD�=���2=�P����<'(F=����l=�廇�3�vO��T=��;��`��Ž�<W�Z��������=�O��E����;���=e�F�o��<�2��g<�m�Á����-��K�=�a���M���<�Sa<�D�=�'�����>�t�:�E��ͼg�'��Mu��?;�|I��!7�W�	=��=������;�V��A��=��>�=c��=8B�=��2=�/P=��,<��=d�����*ې<F�<A{S=g���!=�� :s7��	=A�����=�on<��Y^�<�;x=S�0<�O$�����j����<)��=�o/���/=|v��P9����i5�=Th==�=���d�;t�X=��:���Eh�������]����<�؁�S��=��!=Zi�=}�0�����F���=���=hN�8`�"���=�fԼ6pؼ�	�<AT)�j��VQZ=�q���<��=N���}0���'>d٪<o��g��<Jn���ߵ<��/�0���'�=+��mM��-D��>7=:�0�|���J���p=�`B=�/�1�)� N�$�?�A�*�G�y���=��Ĝ���oL<2=�j����p=���|a���<`���ڼD}'�%���s�������w��"���{<��>?Q��n�<�0�=�9)=5>�"�='�<�W=퓧=�Q��zL���)=���=Sߩ� =N�=��>x�;���^�p����r�=	�ĽF#����/����=���=�n=�ل<���<�ټ�=�ӏ�AK#���X=E��<{ټ� �O�t�>�m=c�=���;��g�����ze.��S�}�;Q� ��k?��󉺘;�p.<�E�=G/;]y�=YL�=�З=�R�����=��@���=1�=]�=I��=r�=h�7�FY�<��A�_��<����{��=q��9qvc���	=�B�G�>�A�[��;$=���=�2=���������<#�"����ē�:�W�=H�<�BU����<��M=D
}�g���2���S����;���=.��=Η���מ=u�m��=i�!<�CV��^��]
;�]��$䍼��ऽ�н�ۯ<�c�:T�J=�b̻��=�F缆i��m�<τ�=s� ��S�<C\�<��=��J=��	=1�<��=,X;R�=�=*_v=�	�G���)=�M����=	��=���<t���Y�=.��=�����౽��ٻAt�<(B=ؕN�,�t�}=��<����h�<�)��I�r�4Z�;Vܻ��.=�(޼F�m�S7�*�%�����t���7=� =����|=�2���)P�,��qe��n��T!.�-2�~˫��܀;eҽ�ʢ����;\I�=ȣ�=�}�<�7;��!�<�o�=�l��{[��-�ﻑ���D�<�^)=,��7��]�>�=��_=���;g��=&ą<,�D���=,���4�ȹ��nϼ�����w�=�ͽ/��3�=��%�.�%<X�������O�Q�	�0Z���<�[k=�g=@I=�Ͻ��ѼҢ��Η<�l<�Y�=o��<�����9=N��<L����Z���k="�=��߼ ��o7�=�Yv��c>=�`����<ŝ�=U���<ڽ�#x�9G;����Q[��� =Ui���O2=�P!=��-�bg=XJs=݀=>���J��=
���1�J�	=I��=  �<�٣=b(��T�;��𼼃������)�P e����s�w=��=�Fֽ�Q>�6����=t��O���1�f�2�r.</�=x{�=ϳ��>�g=BI�����e���S�ҵ{�R/=���<G:�;�;�<�	=�)=�������(<2�ֽD(� &�<�+�y��<��d<b e=�-�=%�/=j�=Z�ҽ�#�;���=y�&>I���=�<t��=�=t�ټj�<�����A=;W�=�;�'�f�m����=m�=��=��<�9��.G��J�;A�hh�<��Q=բ׼|�a=UB=W���lx�;�4>V��<���=�f����M�s���=��ܼ��F=�Y{�S�=�	��
��=w���3[��i�*�jLd:��*=c����%����A�����Z/�=r��x7h;c㑼��=���=�0��*�=M���쨽P}�ePB<�铼"=�B�<QR���A�<�i;=��=]��<Rp�;@K����Y=��Pz���c��	��<
\�IpN=K�=>֎�={��%�l=-`=�G <�
��V3��+��26��{t_�,���P�=�f�<�G�=���<��=|B���	��b��=�r=�4��D��:<����<<}�<��8=p,�,��<o8�<@�]�oћ;���;M�<"@K�9F2=��n�ܾ��������=�=�C�=�s�=��<z�m=pH�=�Wh=�4=Nl�mvz=#�=`-�=rZ<p�= �>=�6.��{9<�=��`�.����t��=�N�.U�w�[����+N�=&R�<c���F�����li��޲�==-&=g�>��_=Z��&�=\����=�|�F<�&:<��^���"=T��<�!���=���$v޻.�-���<�FݼD�˹�?��3�o���;�ܥ���=vH��oM
=��}<���=�.�;�̉=0@�<?���/��<E�=��W=I��<_~�:��\�]��.#�Il)<�U5�&>��a���+;�g.<͕(����!�=��ٽ���r�a=��=��#��g�=H�O�=g��U���d<A�k<'m��d��
=3"<C��=�?==݉��)̼���J��E��=,�6��G������3��LC<r��>�;*|:�%@J�1�3=��=d��=�����<%3�<I��=�ć�;��=��@T�<�R��=��4<b�L=}==Imq�U4�=�V������c+=��<�ؔ<*z5<������<�߫�u!�=�u��T=�2�=�=������/< �>Vb	<��ȼ$�<�\$��%=�y;<����>�����=W̗���3=&U�;魶=��=	]߽��=[}뼘y�<��9=�=�8ֽ�����I�:6Cq=~�N��i�< >'�<��:��c�=���;w����	;CFI�f���- {=�B�=zD[<l%�=!n�|E=�>��=4D<����g)�J@�=�~�=<���͉=p��<6Q��aνU�< ��((�=�F=�)����A&�<��=c�=��<���< u{�uɼ<�ɷ��ú=p���t�==�=�kS��/<4K�<u�.=�ى�Oh?�|;h=���x�<�9��=no����(�B��W<ś/=�[<.�<9���M#T�f�A�C8L��Q<l�a=�`=<]<���W�;��
��Ձ��-�:����y2=x��<)���÷�����LX��叼f'��͙��S>&����=�=�=a�=��=��=_=��;8�M=�h�>_4=������=W)�=ސp���=���=uۧ=R��:�ҋ����&��=�J_<t��ɫ=R�=�5�od�=�<L=�f�<�=n� =��b�y=��޼/h�=/y|=�+h�q����"={�J��T�=>**� \8=��<=��-<dQ	��D0<��/<�Խ�P�=_��w-���x��D�%=�Z�=��<�O2����<��P=@���H!=�_=���ʊټ✑=���=���O�<z����w=2�1�I ���<��=�?�=:ʣ=*����-�<�*i=#�����j��<��̽3�K���s=S_}�L0��B�Z=8]�<�*�=�ͼT�r<V�=/zH;�(�=U`��i鉽�����4=�ZF��t!�kw�����<��m<Q/��$�+�蒈�r�p�_�=Oe�e���J���gy��ѧ���=����c�=�Tv=H7=��n��ۻ�T�@	#�đ?=Y]n�y�=��ݻl>�<�+�=�9s��?�=����i�=J�=�#c=���]r=���=yf�g����.�=q�q=n��<��ؽ�ɺ�n�<����5�^<�����Tq�) ����=���=g�S=�=�I��t��-�=x!�=��=��D���<u'<�<��6�!gʽ�=v>�1>=��=@�ԼU�=��p<,Jy���ռ����}�<9��=�I=������>?ɐ=�(��ٷ����jB���a�闗=�C�=�z=؎=�`�="�=�`�=�)��j=.X�=�z=u�>�9�W��3��=��Ҽ8���Y=�=qW��U��!�;�7j���<�͕=��=�R��<������=�w�=)Wb=�=	�=�<���;��=��<_.�=P�
��好���,�����2�E%�=�wO��V�����Z�0=ջ����l����.�s7<Pp�=��e=���$`�<b������<����Y�<�.���=4�X�/�<;b���=c%=��<J�=�-���A˽�{=��==��s<��C���9>��`=J-�<\NR��AżABY=<Ɏ�="�=;3O���;�ߖ��������=�1�<�^��2�=x~�=/�	��eC= �ؽ36���2���>�i�:��3�(@�=��t�&X�=��0<���=�h�JQ$<h�*<�"�<vT��<p���vX=0q�<m));ņ��>�
��C�=�������<�Xs�{�=L�>�'ɼ�i=�<���vW=�v�=d;�������=#�B�S�����>��;���;�v�ֽ։�!Yt>�w�K(�=��<�
<���ed>�>�UT��<T�N>v �"���v�Y��;�=�:�=iR�����=_����P=A���a=ƻ�x�>�/u�=p�d<}�p=��t=JD
=$R�=�LH��0R>r/*��3; }5�<..>LC>��="r���G�=��>���=ޓ�=*��=U�k<j�$:��C=�K>2�n<8�I�[C;	���y���=��5����<����3���,=ֆ�=)ג������<(+�<���<�#[<R��=���<�3���\ ����V�a����=�������mB�P��`�9`�<\>�)�=�\=I�<nh�$8�:D`2�?J<wmw�i���2�c�����:�U��=O<����7~�I_�8H�x������$�<��$=PI���<=$U �O��= re;"��j�:S�5�
��=���ǣ�K��G�; g+:g���2�=6���k>�n������={�9��/��̽EL�&R@= K >�9o�� =H3�<���'4���=H��<6���8;0�=~�/��0�=r���9����к?�<ij=�n=6��=��J�W$�df�=W�;�eѽ����#F-<���Dդ=W`��5ԻKt�=&"�p��<�,g=v;�����2^߽�Qɽ��/>D���@߽	bv; r�<���#���h�}�<��>0ɷ�� ;`���Z�����<�ܽ!�x����=Ȟ+��S$��6>Y�=�Y�����:޼Ͻ6��� ݑ= ��&�<5��~_=�n�<[a�=���=F�'��IW���I=4%˽7 �;�<[�=Ќ��~ 3�$�>��#�(;[ ;I�n��K;Q�=s�:=Y;��?S�<�X��rf��H�=Z���'<�91���lϯ�LՎ<�zU=.}����Z�[�H�f=�5�i^�=$���5q��罽��U�+���Y^z���=���=[�Ͻ$�O=�,�Dt=n�ϻ�Y㼱�y��2�<�uM��P��Ԏ�sN���PO�д�=�=�p=t:��
��B�=(=�7û*ی��|����$�F��
1M�ඎ�5��=���<�
1���<��ڻT= =��>I�1��<������=����ge�<A}����<���;�鏽��<U#=�<7Ƚfh��򓍽E"</���[�<_�i��[�<4�_;���	k=�%B��5��S��p;�>��=���	C�<r�k=�=�s8��E���c�=+�ɽŴ�ULB=�c�<pe=�L�=�9���c�Pgн�wp�Kۆ=_��7��&+)=��m:��=&�P��Z-<ɠ�<�Y���:�<�4���je�[���_�;U�μ&g��6��Y���;ߒ���*b�;����=ֽF�p�=�S���=9��=w?ռᕈ�2��=0l����-<��2H=�z��+-����!�L;=��Y<���Q���<%��<G�4=��,= �-������=Q���<��b=�>@�5�;=���=d<����1��H�=\�<H���ʭ�=�e~=C7w=ὛN��6=��q=f�g��G;��v�''<���=�=��3=�4��ǐ[=xk��x�L�g�N�=�q��_4=��q=�-��m�j~<Y� �ás=?�x��<��	�J��<w��=;�y�e�����0�I#�������<�f?��%�=.�=��+,=�f'�[H�r����KݽrX
>���=y.�=���;�W�=���=� �<,�v=�i=dF���'�O=[Iu��N�;S,�<ԓ=�'�~�=�F<ӡ`=�Ä=Ȼ>=��D�`�q=d}�=�Dj=Ta����=��;zr>��=5v5�E���1��i�=�%��삽B��.�=�����/J�;�L=7�=���;���<�Mn�xͽU��<��3�t���<�=`>��=*�=�粽nǽ_ԓ���c=no�=A�����H�e--=�焽�'�Y=S.�;ű=+�C=o���������=3��k��<j�>�s�7=u��߈��z�D����<��b=�DZ=��	���q=�IV=b 輭zu<Q׮���i�w��</���W� ��lü�s�����&e��BȽ�3�=�E�=��O����	�=�E��!Z�yK�=y�=��=��n��"?�=��3�	��<�޶�裙��S|�D�(>�h�=�!9�g꼬��<��B=5ؼ���=Ï�;b�p��F=�-1=�8	=�b̻��<=Y�F�BxP<;:"=��5=?s=^��Ac�@������=Aα���	��mM�\�ּ�g�=.,�<�ɕ���T=z���a�<��7=$Kn=@�C�����jq��<�s�_O���E��'Ͻ�Gɽ�i���=���.b=O��=J2I:�N�<S�P�qƽ�����r9�m��=ϵv�̓���<��
<6��gc/���������,2�=���< F�=��+����<�M�$�l��K=>�a=u1��q��<��4��x=���a��G.��F���3��[���D���>�|^����¬T� �R��i�<�_<�+�=���1��<�Y�^2��H�:�>�<�Ю<����,�=j,>Ň�=z��`��E�=��0�Т�]Q����*�>�2<g��SC�=���R�	�t�W> �<�3���>�I�4@�9"m�<�%k=�ᨽtr4�)B齑<>Bb�=�t���B>|;'�{�=c	�d���^:=!�����;$S<>��=�Q��c�
��?R�xv<qy�;X[p=_6�<Q�������=��=�P3�>U���=cTs<�<���;,��=v.v=:�=��B<��M=EӼ��j��T=�qJ�R���U�>����<�V���!=�c����=�`�=~
�8ح���»، >{���a:��h���X���yA�~�+=��;5���i�;T�.C=N�a=�s�����C<�iԼ��=
��x�E���7<`=I�4�1&k��u[�����IZ<Y=Ψ�:����V�=�e��P=s�=�3�'=���;� =�b��_�<%1�/���->��!=f/��`=��%=A�� H��lD=��c=歟��&;*)=��)��=M6�<�$�=��C`��7ņ=ω�ꩻ<@����	���>��-=����PVu��s���oE��2�=^�:���<H��B�	����4��i���a�5�[6h<�>����򕂽�< �����=��=��<�5�=�qͼ乮��ߖ�r�{==�P���7��� �7����=�+x;A�(>���WϜ��0"=��S�ea�=��Ｔ��\l|���=h�����s�5<Xn�dRϽ�ǚ=��=`�Ӽ�`�=;~߼⽻=?��ˬ�<1ns<bSg���;�J����Žqx�=�"P���ϻ���S5�;6|"<��W=\��MƱ=}j��Qڽ�nk�|s=�>��⑽���8�z���>Ʌ�J�#�˳ȼ(�=��λ6hF�Wj��j����Y<���=�"L�,�	��d���Hż7]=s�μ}�/=��0��eC�p���t>�1=O��<X�|�oټ�V8�><a=2 �<x<�<��C=o:��e�;�Z�
��<�B>�4�<�����
��≽�'⼳\d���'Y�<�,¼�3�:Iv���V=��i!P;�3�<-.w=��;�a�?�!@=f��ؘ��7�\: ��<�M���F�;��w=+災9�#��<��~�93�3�n��q�<�'x=�?�I����?^����;�/r�)f<=��z�<�k2=�k��M�\�J�v���9����:�<(d=�".<�Jּ��2<����t$=%����Gz=�=zBU=MF<��<�7<�;=�3@�|�K;	��;;��=N�Y��ϼC�j>��B��x�Ѽ~���N<(ć<7*�;=ى=���<	l�j6;=[�h=�� =V�=��=_�<6><+���h6��!d��[���&���*�b=X�E��b3;���b\�S�1<Q�G�9S"�����</T�<��@=&�ܼp�ӽ��</!��Wf�<L�'=��*>]k��,�pA <��m��7>�@>�^�ֈ��l>5������=3��<\�}�@L&�\�=��3>��+�����<)f�=�N�=����(7�=kB�t�ټ�:{�j׻��!>w�&=Ԑ0=��*���<OJ,>�(?>a�"��5?��?�=��~����<��<� �E�`=hE/���S�>������Ѝ½=�=��>:0
�R&<��s<4�e>�=�e���y��K�<��<3��<x_"�j��=u�_���4=�u�<̵���ѼIM�u��<1I����7<5�;�U�<����zr��XؼoE	�u�$=��=�j<�����v9�i���B����F<���I6�=i.�u��<HI�Y����&���։��w�;��k���B��	�<p�T���Ř��*$9�~|����򽽗��P���&�0��I�[�H�'!T<}��F=��=��;"��zu���7:��0���<�F)=�Ϋ=�%�=�~ɼ����{������Ym�=�=���<�؁��򓽟�r=�#���ђ<�e<$==j���E��e��<5�<e"!�ݶ��^�==��B���<P��=�q��a�<)ॽѢ�<̢�C|<m�>�g�=���I�����U='V�Z�μkDe=�G��ý�@>�G���@�=��;
�yr���,�(���%��84@����=�.'�%��/��<�{;�۳<:h�<$j��!'<"1Ƚ<c�*���=~���tS=�s<w�W�=�
=�)ἛaK���,<%q�=�7<��h<�!=���=ȵ=vq(=�=��=��G����=�;��6��w�=�YT�}Q�<;�<�����S>��T=LQ�<[�=ýY�=6{�;(�=����r�>��G�n����~3=�ix���=��,<�~O��n�:���<� ���w�<3�N���8��!�=� =�̃=�7��}@=�XX<�x��!�
=��7��](=���=��x��@e;)�#�.V��2���;�<����'k=�=��<�.����)���c��Ǎ��r�=�*3=K�� �;}�H>����<=�<��<��=_V"�	� =s6�=�W:�V��	���F��˭�=<  �����,���㽂f���{
���>O���nݼ\n�<eִ<��)���3>��]Y�|䪼E�㼶񪽐B�<�¼Ӗ��(�O���`*K>��
����s+����m�=:=No;������<r�;��(νy�1<�~���A��CV�=r�=��=�=���<�_�PtQ<fα=������佼zܛ=2�}���<�s�]�]�T��SY��l�<�D=�F��=n�I=��;���b��<NQ?���l������<-�<'ڼ��T;&�=WI=�t=��<���<���蓻;�P==��;��0<��<uc����'=��<�����b�l����
=�<�&==Z ����.�-=�������<4!�H��<��ʽ6<"���>��<\v���2��X��·):�;�z�G=2\ <-���(�w�rj���)�O��=W�T=Bנ�?���������E��;�=d뗼0��{��=I���[<ÄS�I�$=F*�=�	A�ٸԽ�c��;���4��=K�
�{=�TѼ�N=C��Z��.Aż�*�=E(�=&%���|^=���ڴZ�hA;�7=��=
��<
/��Q�=ȴ�=�W=CV;=�O��&ܻfa(�/N=�H�;瀮���6<ܻ�<� �=ΰ?<9�=��<�3�=J�Q;| ֺP�=O
�<�=ܨI=��=r�=��"�_��;{s"=�{>C����\=&=�=e:4=7H!�b�K���#=��
=���<kF=ۇs���=�����<Yp<�=��,�?p=k;Q<�S����<�`�=�}"���<�ˀ��~U���B=��S;\c=C�7�z�=�>>���φ�,����Zؽ�;�=p�]=��@=�q�}⁼]��<�s�=-���ه=��p=kk��ed�<��<�;�<�-'<��0=�Q�U��=�
t=��<S�E2����$�A��<.�*;��q=.�=��j��u���
<M��=�0=V����;���4\�<�#=tb�<�n�=��=��p����9H�<�Ŝ=,�]�D���d���8�=�KP�lR+=z����u=j���N��<b�P=]���<��O<*=�ø��=� {��m���Ѽ2��=}]r9h�= =��I4J=͖X��� �=��;�;>"<Z;�Dм~*:=*9�=goW=�f	>�g�<Ð��$�@=Ba<=L����Iv����=����7uV=�y>�{�<59�=�3���*��{i=��<�u�7�0��!�<l>�[�=z�=�)�=6p�)q@=ף�@��=`�-���P��>����<iK�἖<�qw���=��޺g�,<*~�M��=Ӟw��F���=��=�IW=���	`�=�\;oyb=��=�h=8�a;b�x<u�\<�ڣ��57<�E=TѼa�;񋕼�U�;&�)=~(�<�=�o����=�ݽ�x��t�;��h�}���]=_�=Ī5�&w������߽w)�=�+�=��@���>�t��=���=h<�C/�(ں<$pH=�H���"��LX��-����=%H���r���=���=��B��0^<{3�h �!�߼hC��xǻŚ�<DL��K�׽C �=��={�;��f��"���=��<JH�<޿<�c}����<ӑm=ǻ�|T`�h�= �ټb�(��ֱ<䡬�����4�"劽�=�� =��<z�|<�A�<1Ž��W�.f�<H�=ar����<�i���C�=NH�=]�=(/K�yhM��=���<Ov��Y���Z-�y:��鳛��z�:c{<bʡ:�����D�ԫ<}
�=d���$��\�=�H6�q��:T��=Q.A��I,<<&�<¼�9�^�\n=�_=>y��;��ڽ��=�x�=5�:��=3	��`�=�A1�!��<q��V4�<q]�<Y5=��Y=�׷=���[��=[���J#=��X�!32�	�D����h��<q^'������Ҡ�<���W�L�S{)��m��].<5�a�]�;�m�;�d�c3ռ��"<O�f<�j==�K���v;Ӕm<����4g=_5���@z���R;>V^���;�h�;����2�=ը<6H��f�=��,�rf��@ѻ9���{˙=����=<)���.�=��6=Ͱ�;��=�?<�`�������g-<��6����:�[�sx=:y�XPֻw��<�?�=��=�Y�����<��=���:-���k���$��+����W�=Pڽ������<&�;����˄<�E�=?��<nB���9p=!��;��ɼ�AH�$�P��+�<އ��A��<Z�����!=��5����<:���j߿=��@�p�[�����g�<-�n=��<v�U<�n麐�P=8@)<�j�v�]=kE�;eBٽt��H����eP=�勼�7�:��=��:=�ǼLd�����$@���<!,��=:e�=ݮ�;�m>V~D���8��=���=��<4`���(�ViO���<��=��n��*>z�%����ۼ=h�=��U<��9���=�&	=W�=|�r=�ֲ;u��3�_=�#�O�?�����`�,=}�	�'f��:=ry�av�����Q�#�h
>jo�<o�U���Y<ۦk�P�=h�<q�<P�X��u�=X+��s�;l�`=5�<����d��=���?�T�Uov�Ơ�<9��:��@=�4h�����k:��_��{��醦�6@S�I�5�{F�T���}*�=���=��<!ac<����|� �{h�<~����R��<���<`�=�yg��<���"
<-�G��� �{��;O�L=�=�a;ػ�<j���e��;P?+=����"<^<BHv�W��Vw.�8��<�9)����-�V���
=�F��qn�=6줽��5�XᎼUp<��ļH�=H�w�?k5=�;���Q�z=�g{<�h|��
=�v�FF�8�Y=w���`�e=qY�����=����Dɼ���;�$G=�T��"�&����;k�c=�w�=�����ẰJu=�<	�vD<<c*~=�����l���U��"#�<t= �_=�R=�2�<�� =�M�=G�g<r=�脽fm����k�<�F=�'�eD����=1��Y�;�B��]< 
��ܒ�<p���\̽z��<�@� .ۼ��O�M�軏�q=!*�;O=UJ��G��?b��*v��42�=��мgL=wrսװ�=Q>��۔9=Q�<K��=tQ����Ҽ/{�tk=������<�P��w�z=˼���=(<�=��c���=�g��<����.�<�F�7�l����<�RX<�;ͽ�_<�:a��<�����ρ����=�	i�S�=M���Շy=�g�4��=,��;!l=�a�,�F�&=�	��^ؼ�%�=�1#=s�H=�K<�>׽�^�=���=&o7�ևH==���<��b��e��81=���O�<n�н�����Ҭ=w�7=�A�M�=�z�=7�=b^�=/N���4��� <>����1Xǽ��R�z�9=��>.+�����Tf=Ⴂ������V">+>d>��+=R5��ʘ=;�+<!~=���v
>A�h��nc���B,=�ǽ���2�f�s�<�s��)3�=]/<�{�=t����HJ<�<�=��q1=7}S=�6A��/ϽE��=��>�#���Q��V�=���=�r�=�^�=�l@�I6=59��O�=]2��h���+=���;Sn��4�B�����������`��<�u�=ͣ�=�J�<4j\�
{�:X>��M��`7��L+V�k=��R���@�o
�<�_T=*����L��=Ro���a�9�@�=�a��4DŻ=�<�d�<$!=i8�*�=���/�=A4�<�ʸ=w�ƽ�Vv;
F��>}<\�=A
�̜7>�6=�i�<��=�~���P�Q��=��D����=h$���А�4���٤B��7�=����gϪ;��+������qb�o�$�Dȝ�]S?=M �=h���>E	�s��{t��B�yϤ<8}i�e�b=�F���t�= 6d=Mr<i����=�s<�CA�mx��M� ����<ź'�r7=w)7�u�R���=^�������ѱ�$�<:)E���4=�я�[|C�|I�B����=h=|]�;�Oi����<���?V�<`�U<3X��Bܼ�+1�����[=?�|��Q���;��Z<$FV�����/����X<~��<�X�=��y<qb��������<����"�<�Y�<;��<D�c�ёR�O؞=���=�)�=�Mܽ�\T��|��A~�����;)��<s�<ZF��ߐ����}�|}Z<J�<��޽��Z��u�=���=ۭ۽�ى�e����[����"�ϊ�Cv���<#�S�ڽ=��<�#>��Y�O��T;�ƱZ=�z��٘�<���<X�(�Y%=%_����=��M=��t=�����'=k
[=:A��j�И<C�*��1(<3p�=N��	Em��<���]���=�2`=�Z<$�ּ�g�l�-��=�l=�c<�q�X�b���
��=$�a�%-�<��<Aļ�s��0RS=���oB�=��н��Լ���;����EL�`j�v�<�c�=#�Ž�A��|M�ɮ.=	�n��:��&=�v�=cXѼ A�=���v�,�;�R�<-����T���\������!4<d��=�����S��!�fվ<a�=)4��>�*)�<Ǘ��i�����O�WC��G<6�f��z�<TF���g��;������;h8 >��">������;E���=����Eb>� ��KW��-��T��9�*=L��>�y�=�u����	>	/>���ud��sн������	�.��
h�{ͼ}�6�~b���R>q�ȼ��;��&>Y�н�h=��>��J�\[�=�s8=a�>D���7����=͸�=�If>@��</�<��ڽ���>gދ����HI]����� =��I�6�>��4�s�۽;�;�s�<�_�<�%��O��=�&���.4�ȑ��l>>����G���w��tmS�	��= �=�A=����|�j=�a�</��<��=�D<�۽�GU�,�P=؏= K��a����<���4(�X�m�6�ҽOU��A��� �O#�=] ���=�躻��>=0c���#�=t蟻�I}��!�=�}=�j�~������<ʦ%>L��N�X��F�=<�=��g=N�Y=%�<m�==����/I�m9��D>����<��=�����<��A=:c�<F�=$��<����c^<Y�<� �p���پ�k�=Wu�=3l]=��w����=�Ǉ��|:��?�=t�d<E���Ƃ��tG=$|=(���j���K�<�2c<�dF�x�<w� �d"/����d<U����=g^ =���z��Տ�<|7��x��+����~�<��=&@�ɭ�=� �oA�=k��=aI�<*���'=�� =fM�=~��=V0�pU1=���ɾ�=%,��=����x�<CAF��;�Q̴=?[�;Ch=��	�EKټؕ��pV��!�=��6|�p"=X޿����=�v�;��@=n}=b��� �=��ռ��1������1ѻ�c=��q��z�1U��Q�=����>=�ߖ<��G�<k�<[�y���=�漩��=`����?�;���<���;��+=���=��;��=C�[���X�c%=B�,=N*=�(�<��=lN���i���p�=�	<�S��=kʡ=����2�<��'��3�:%}�<���Y�V������2���=����^�<�3�=��н\#���;7>�x=��'�x���9$�e���$��=Q͂��n��J����=�����<�uU�tk��_�������`.���Ƽ�/��ϴ!����ֽ+s&��?9=��)��F�e6�AI<f�< s:<I=�*y��ꅽ��<��TN=Zp�=�x=���.Qڽ:�*>�s<��<������=\����ͼl>���<,�(�@J;���A=aw�=8^����� 1��;Xݼ�w�=ZT�=���<6��=x�H���=/��;V\�=�B=������=rƫ��q��5IW=�L�=����y�v{=���=#f�������W�=�Լ�:�Z>Խc�<h���ջ����ґ�c�<Z�=���?��<�=�Z㼁�d=J[�*���6�=�\������l�<��=�F�=a��f2=i�;��~�S�����O�`M��?�<^1=� �AD�� ���9>�Nռe<�D)���I�=�=ؼ���<R|7=@1��V��T��<�D��'�=��u=�{�=�E�A=�Yƽ�%����'���=��r;{e��
��=u\3�c�<��Q��;��=S�Խf+���1I<f3 �����#�R�t:�#=ù�=Nu�����������[a��
�=P�=�>~=No�<�dнO(���H��=/s�=��:�6���B=�RC��Ÿ9/a޽����lu�=�"�=�-�r�'=֮�<��=���=�Ј��C�;��<����'�rO�=��=�U聽bӶ�Ƶ��lR�~}�
}�;���Bؽ��=B�ֽ�u��B괼K�e=h�;_t��6=�����\Y�����w=xa�=L���ս���<WN�VYS=wx���V�ĸ�=���=�.\��=߷<�dἼ��;���;���=PW�q&<���(��=���=) ��;%�w�Ľ�RT=�i =��һT�����D�n�y=7���4=p�]<*I��8�M�T�|�SX���D�7e6��q��!J=9�<Y�#>Cp|�_����h���E�6���/�=%���ݱ���̺g�ڽ婩����=\��=���!����a	>�������Q���=�7=�$z��n�)�B�iG����ὖ��=�D��^�>2�>/W�����=7�=W�ὢ�N�$4�1r�=�kݽ��Ҽ>��D(=t�>	&ǼP�=��F=Y�'>ٿ
���I=����8Sż��=�(��`�<4J�<���� >zo"=��=(H�=�eI<�:����}��hl�=���<���ZP�<꼝�����g��=��=��_<r�������d{��(�=���=��;N�����=�?$�~�m<Z7�<��=�J��G�ݼ$=�;i��T���О=��D��d�=�VK;+�=U���Ƽ>�-��=A�<-���w�>��Z=�����k����������;�����*�=A1��.<�>=`�ƽ6�<�f�΅)>�F��{�
�{=���ռ��=oWT���?= Nh���1�&k<����ǂ�;�&M<�敼�+�<md��=T�=f�=��N<͟>�i=��B=�/0�L��<d4��0�����z�7=ۇ	=Tu�<��½�zƼ�D��XB<�Ԍ��2�9��=nd�ݻ�B[z=q�x=q��r�<����&��<o'Ž?Z=@<�uB=Ȝ�;{��	ԼH��������M�d�im>=9�a�ջd=(��<v���P����彉@Y����=�涻-�R��I�1�뻳Ə=����P�n=��=��=G��<2-<1c���򼗝�=�ژ=�9�=-=�]�=�lB����<��=X�=�]	���?���]��`�=��:G�&=�x���=޵%��Mg���Y�A����=< ��G=nn�=t�>�t�j�u��5��,nS�[0���0���c=�Z��!5�Ǘ�:2=�$�=�))=,�#���<o��=�b���<����y}�=�n*�WP_=6Ƚ}������<=N�<3N�<��7��T�;MH�:X(ļ�=R��<�q��f~����=��D;�������<u���iN=�����Ώ����;��<#�=F:�;U=���[#�E2=�}n<�u��.ɽ#|%=V�<WU�==�
�<5�G=�M�<����Y=,8��K
0=d�r��$Q��Q�\�B����)��<�ǻ���ż�8�=��=	y0���;��?<�o;R-�=4�����<Aޒ����=�'��c<��<���DL�(�<�J���=n��%�$��j<yU�<s�%���Z=��=p_޽	��N�	�9˜=�b�;����T#=��<
�!��嵽��<?!n=?>b=�w�d�=M�R��}��g�=��<�=$��<]���?�I�,B �H��=�@��Ď{=��=�μ��<A(=9��+8;���=G�e��49�~;�QCW�����"G<P54��P��iI���0�XT�= =�&!�����x;�Vz�;Y�= l/�8���$=ɰ(=����@��N"�%<vJ��0��X�;D4<(㸼DW=�5��UƸ�q���ah��1=�d����)��W�������6�o+�=iZ=�o�1��=Ȼ��9���<p��=�6�=A�:�l:�����={�^�K����gJ����=�ݽ<&�	����y�=k��< J��I�K=��ּN=5=�v��d�>2X�-e=z�Gk�<Ԕ�=T1���T׼p6�����=��ί�<g�G=��N=�tU�&a�=��Y<�G޻�E��C=Ӿ�<0���<�᲼I
�T�\=�z�=T�=���=;Q ���a=�>&�V�ݻ:hY<;ɣ�š<�i<0���ɬ���d;�'�y�A=B�T=�-3�Ԙƽ��l��p0����}�������Y&�Q[�<M�Z<a�<+	=G��R%���!�<�f�=}8�<Ӽ�D���"�cṻQ�<� 7�9#P=�-2<��=��;�G<Ԕ]=��%�<�q����<�HE:�H�<�ߪ�ⴑ=�Z�b-��ZN���`=�6�Q�<<��=}0=���=˼�=�a�=�K=H���>g�=��	��_?X<����`���B��4�;r�D=����J=1�iz=��*�<��;��,�ڐ�=�L:�cY-�z��<ʌ�=\j���K=������ڽ�3��Q�-�v���H+����4�ͽ�LG�s��� �=������i��V=a)=l��o�i:8=���<Bu����˽ �b�r*�MQO=QI�<�<1�;[_=?ڍ<��'<�)���[�8����<�ؼn��2�=Ŋ�9� ռ��=l[x�(��=zx�#1=��4��!˼�I�;xG�<hu��^���m��=]���Y���;��m=סk�Щ{<�x�<�һ�v��VE�=Q�d�Z���ϒ <\1�� d�=�g�M�����=#<�����������;K�<�ػn0��'�<a����,=���<l��<��<qP�� ^��&�=�:�;��=4^U;;�>#��<c�=�6=�肽�xɽ��=��;.�2=L�<RW =%6=�*";TS�=�|R<@�K=��7̎��</=-��=f0�=������`2=�H�<,�=�4Ľ� [<_�t��.��x��,"���M�R+i�j��=��<�����$�<���>�4�E��<?�C;��;���=�|����<ܼ���<��ּ���=Wȧ<�;�<�B�=��[��<�G!\=���;>�Q�S�G�T�>��.=�h=��_=L��=.[=u��<���=m,��p�X=�=�-=��;t�=�;���<	A�=c��<������I=rH�����\=}�{<U\���b��n�=n��<�Z�@)�<�&>�P=�E��V<�Ƚn4ѹ��=�������=�������KN:;�?=�2����gʴ���<9�=��j<��˽�����t=�q<�%:������»��=��=+4��uŁ�Am?=�t<�V��\�#�<I~N<�@=c,<	ż^�=��<�=evh�m&-��=�;��=���<0/��#�&v�<9f�=�AR=
��u >Id5��9�xӡ=�3�;�˞�p4�'	�<Ś����B=[=0�; A+8�t�<�a��M�����Lm�=����K�;�<�/=D����=�h�O��=Xƃ��p�:�؝<_o���ъ��D!=H�=1�=�(<ˊ �绡=L=l$J=a0;`���=�A!��<�k2��u�<��V=r}=OZ���;�=��<!٬=���8�Ƚ�Q�:du�<�I<�޻c0=�%	��� ���>�0�с=��+=_���y��Iy����=1ԋ<��^�@=2jt={M7�'�J�����f�=1c�~�����)<d����8� a����O�H=�c�<�9���p���N����6=�
J=��N�{4�=�Ԑ<�<Ƿ7��=�<�~�<��M����>=!�@=�=�=q�����G��~B��.=+k��0 ��[��<���=���<�u<���n�<��n=����>�~�=��@�f>�;�j��3����v!=�	��j5�ٕ�=;�a7l�U������=��<�r$�=?�=5��=Ζ�=�>=����V�'n����<~î�ê��9��ƽZ��.���qS<�W�Q��,V;>#���������~<��,����;�Tc��Ń<l�<�3n=J�)`�<6��;������v<yZ�8]�<���:d�=���<�ỽgcQ���=�����h=0���^O�< �������$�<��=�֘=8Z/���f<�ʧ<��<µ>��t�ߪ���h$<�~޼�V_=��=��=����y�9{�=.����=B��;G�=$&�<���=> <��M�ʟ���!�<=����b���=Qe�=��ջ,��8����<��h�=�,=�f�."��Ƹ�L��<��<�\�=%��<C7���u<�2p���>�I�;���ht=~6�<��1=��|��#��[JP<�l;�Mx��+a=E�0==�@���Q���=)�(���< ~ >0=E���%>�=W���V=�'����8��;��=Ա4�M=�E��=�0�cC���#�;r>�=�O�<���=�1�=��¼�1�5�<鶞��ד=}����z\����; �ֹ����{<d����V=���]+�[���t�<�QC<et��,��96��;�KI���<P�k�=�>4l��gn=���<Ł����=dr�=�85=��׹���!%8;*Ң����0����n��N����]�<F%�����>;??���Rf=�ٽ�1�%��>� ��iB�98��Ĭ����t�>�F=�Z��N-=�;>���
���W㉽���={s�<ӿ\�'���Q���;xV�<=��=���_��=�K�>�.����=ف >lNƽ~}=����%�>P�x� "�<q�=ܢn�MW�>d��=;��=��{�=�>%)���m����B��M��B�a=ƞ}�wB�=�2�<𸏽�� =�5=����AѨ��>��H�Q���t��z0�<ܓ��x`��E�ȼu�w=&/�<�ߒ=��`�6�=Q-n�x���ΉE<Y���U�F3ֽS�=�|�=��I�y��<='=�׹�\�G=>��o��"��x�[�K�U���	=H�
=�:J<�f�#o=1�8�Ym�;��_=D�d�t��=F_̼���=�wD���$,�<��>�c�#=xD<־e=Մ<I�[<R«�%����3u=�^Ž[�=V]4=��=tNG=���=.�A=��>���:O��c�D��r���O�:�q��y�<jY���,�c#}=����<��s=���<w��<U�=?+=���� +�P��<�*=U$&=z�������䯻Х����c=\e�a��<KG�ز>�I���� <r����%x=�Yd�^<@=����`�<���=^�=�+�	2N=ǀ�=yb�� 轉DP=�q��y��=�`R��N<�:C<붘=k��7�{�Q<�`�=��E{�<1��?y׽������U���7=!�=i��=��~��-F���+�s�������x�;3�o!�� ��l�ֻم3�F��=!މ=9����XU>���u<*A��Ყ�
�"� �e�<�#z��飼�b�Z>�-ҽ��<���=M���U={=�f�=�G��掽;���*O�=���
�5={a_�Z�<Rg�=����形�U��Ȭ=T1�(����F��d[<A�=X۰���7>
��������r��Ř��A��dT�� =Pt�����
=7,=xPJ�ҁ�< ��A�����k�=�ӹ�� ��ϻE;���As���<=ir�=�GI�EI�\X�==衼FK=�G���̨<e�;uf�3�*���<2���R<���;� ��#�<n1�=e���D)����z`�<?�W=H�K����=������W�/Υ=�x�Y >��=Z�>�+�<�=�M�=39=�f���}�5�0��<6@��2���n��h�>{�t<5�=4#�<��:=v���� =�h�j佽��i� �\=�8+<<Jn�68�>V�<&6�<�3���<��+Ux=u`{�f	ۺ�a�:.�<�zN=>��h�<�*�<��v;q�=�T(<�Ӽ�X�;ȱѼG�;�h-<���=�>р�=N9�=�=����4<J��;�v����:�
-<�(I���(;Zp������<���_���^=�=ETм#��<1=жF�<"�<�j�=���}��҆{� a����=��=�>b���K=�kk=b������!.a<ʥg=�C�=�u=��<�6����¸&�F�=d�~=e`�/�=G�ڽ1 ����<�Ś=
,5�}������=�{��;��������r�� ^�=�S��ա�Ȓ=Ɉ��<H`,��1?��7ͼFH{=�����tW=/�׽w��=�(=�v��c�=C}<�)\s�����F=�VO=-�=��#������=3��<��=m&�;��<ݬ�<��=��ܽ�婼���<B)=L�g�$�q=��;��=TC켕+���@�o,�VS<;�E=�.2;$<�g=��=���ud��7�=-d���%W=�7�Wk4=+#�<����Q��t�rF�#��;�b�{�˼�5���4���I�@�k��m8=�u���<���:&�_=-A���e��o����0d=�=�%�4X��M��=a3�4�}=����'��<��!>+�뽗��<ற�E5�;Ζ=���<�w;P�t:���=M��?�P="���y�a��)=<=_,��ޠ�<0)G���=�l&��#d8�q;V���wۢ<HR";�`����=o̭;�N�<�e�����<��3�{�>��s=�M�<<��� ��C�;={�ͼ�L=\)'=x���s޼�.%���I��cO�P4���	�ؠ��?�52m=Z}�<�bW��g�=�:X=<�
<%'�<�x\����=is���&�A.���=�=�y-��2�<̴�O��EǦ�����8ؽF�������td=��==ߥz;��<�Tw�j�r��9<<>ݼ�)G=�-�=Ib@�'��<�x�<3�0�
�½�nμpc�<v���'!�=z2=���<p�>���}�����0μm(�<H��͓=cﯽr<Qo��s�<;���kf�)��ɫ�<E�~��ϑ�WB� =D�=��=�ؽ,nż-g��9\�<�$�9�k:M�T=��D<%���=���=W �=}�=�0��=�4<��=�#f=æ4�E~�������6<��'��e���N=U�;�'t�"'�<���l�=W��;�����">=��<��ɽ��<ɝ�=E�<H��=8�<���=O{��7%�<����gG�=b�!<�>%�b�W�2=����v:<!�R�g�󄼽9�=���O:`�p^��w��".��o��WM5=�ǻ=�R =���U�(;�*���>M��R���?�*P�<A����E=.�A����pѢ<���� ͻWV<0�<������i<�q���e-���b=��9���<$�Q=T���b��=�k�<��Y=���<�q��=8>�};��n�Q�C=`t?���$�λt�:=xY�<l�=�E;����>��<�����B��S��<�>B��<=QG`=6�[=P����<�B��]�;�P�"z�,^h�('=}=��.��+==Z��<��ǽ��==U�=��V<1�V���=9��=08J�	ݏ=9��<Q�˼k�:�l8;���<u�>4ڠ=W�w=d�?=�;";5%;����l��;g�=>��=��E�f�y���\=�n<�|=4穽�U���gր=y;�<gZ���a==1
>o9.<^��<�e�=�_=�N9��}ٽ���=C'�=b�=SQнA1o=�cY������8=�<�|=t �=�8�<Q��;L�<�ۊ�V:Z�rD���y��$?<�-�����=@x��s�<�5�=�-�;���=���f����(:�=��=wy�<gkK���=���u��=�1��6�ռ%��j����y�3P=��w2���=-�=0$=��=f�U���(=�c�=q������$>S�e=vÙ�(�:�"a=+,~��e�=�{��">��J����=u���Ϛ=��=�bx=yu�Su=�[�=g�=��F����@=����XQ�<��5�Eyn�i��쎡<1N�{ۼ<\�<�S�M���'�=k��q��:���;o�=�$=m�E=^*�=��ۼ��{�ё�=-Dh����E�^<�K��i=��=��%��ϡ<<u���=�枾�����E��B�=�-=1����=D�U<Hh���0�<�r��ī�=�;I=USI��H�<�d���<r=/GX=rE~=I`����|:	9�i(=�v�=��޼'k��G:���O㻵I�<��;��/=*F���*�=y���>̼�`�<|ș�� ������u���2�p��<e��(�^<��y<}8"�r��=�E���<��<TǕ:h�P<Oh�<�B=+Kl��Z�����=�u=�W����r�����;z=��˼�m�<�*>��:<?j=@��<�j=c�����<Ңm=m��<,JL�L�=) �=^ƾ��?���{�x�=��=˲C=�ÿ��T*=��8�0���\=�^>�f`=.5=��u��Z#��W���`=�>`l�;�V��	�ϼ�k�ѤսCyѻ�lƽD�=l5�<�(=B�^�n<����Ž`�a=�j��#�=�]��?��<�J�����=���=��= C&�w!
��*N=Qv�y�<(���=��<ީm�Aq�=�s�D����<k �<~���ˏ�<��H��S�u�Y�*<�ڽ�WG>n�=?i��漜ຽ�L�����=��;�T�=�.p=~��$�<��>��b=7�c<�	��:�=�hϽ�Bi=�bN�LzR=�[̼8ƺ=Anw�_0�K��T�U�}^3��>��?���K>v4�4�=Ջ���3�:8�i�ݥϽ�8>�]�<Q��=�N����-F�>�'�=����N��9���=�G�5�Ƚ6�m= ˬ���"�K;��5'��Z���=�=m��<���<�ލ=	�=3X��p$�fu=;�>��>��ټt%���Ϡ=V��)o�s6m�M���'��<��WϽL��=_�=U[=S��t�=�--��^Q<���������f����:��3�V5q=�༦��(%d��߼;4*�BiU�{��=�)�<�3���^=���W9[����������<����Ȉ >�`m=�]���Y1�C���i�_<�<�=���<�68�r�f��ּ)���c~ҽ�f���N�<�<%���YL�<B�W=�`���T�\Mw�+W���eo> �Q=��=�[=1�=ᏺ<���8���;�J�<�=�63=(~9��>`�>���kd�������=���xF=���<���7i�=�'�<��=2O�AQ=���3r=��q�M�I:0�s��\b�Q����;������!=���3�۽r�;��� �m�>�|�NE���< �=��8<��r@;��C�G=C��=OM��X[�R��<n���os�<�^��Ȍ��e�R�I�9
<��伡
d�9�M=�X�<���5Y=�N=}ǣ;�λ*�ν���=@5���F���E=D����	GD���/��]J�$�Q=:��t�����=Z{�<D׽�i4���N��X��X?,���9<֗�<�=�������s!��>#��*����:s�}=��};���=�򽻺����|c>�i�,�M�4=�`>=��^�.�"=F��=r������<[d���缶=E%�=�X4���<�P�<TԻл>�������<F�[=�׬<՞�=��;���=��.>8	o=͘w=n� =�a�=���;�*>�>�@��t�%�$>���<��U�h�=f&�"���'�=!k�}���3½�̛=#�5��[^=d!�=D��=��b<��=k伽�a<�I�=do�(>y����)c�"����H�=���=�
�<ճ�X"<�޼=��E=n�3<�w�:�C=�T�ڕX��i��+�=P5�=28
6StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_5/conv1dConv2DSStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������@�*
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
:����������@2S
QStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_5/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_5/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpConst*
_output_shapes
:@*
dtype0*�
value�B�@*��m�;�Ƹ�������������1����%��~*��9��ں�ϋ<�( �\Oi;�P���F�� ιI<w��|��ns�=��K����>!<�N��4���S"�H�;@C�tU;��2������7�k'�b%68U�|��3�^�����A;|wy�Mo:���=�6}��W2��H�3H�߹üZ~�<\���'�9�Y�A�+�&���Y<��l:�ht��Sˠ<��<�a6�Q=;e�<��M��h�����)q̻�E�:2;
9StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_5/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_5/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������@2,
*StatefulPartitionedCall/mnist/fc_5/BiasAdd�
'StatefulPartitionedCall/mnist/fc_5/ReluRelu3StatefulPartitionedCall/mnist/fc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2)
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
:����������@26
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
:���������@�2Q
OStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizerف
6StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1Const*&
_output_shapes
:@@*
dtype0*��
value��B��@@*���=��<7�ϼt��'�=�y(>q�T�8�=B49<�)Z���߼�Wh;��(����<L޽3����Xa��0���<����=�/=җ����=[r/=��>�\5)��-=�b���h;�N=�F	����#m2���(=�
�g�P�#��I�+��ay���P=7�_=��=��B�Vg�.����μ�=7~���<���(� =nֽ	�%=�*���\K������ș<��=�鐼X�V���=i�r�׃��-�<����';������p��
���<h�<��=�U;�7[��-��1�����I=�喽�s���Լ�Y�<ԃX;^!�� "�UT��'�)=�茼�`��۴Ѽ2��;j��M谻0v�<�&�<��������=�ia=}��=��;=	kL=y಻c(�<2�Ӽ��=��-:S�u�	��I�=��j��]o�E����J?�XGW����;�/=�4=��<�L	�H�{���<���<��<����;1�� Q�}��=�)�=�IJ<��]����;�Fk=��{=^E��2�<�=�u��3x�\�����T�=�.)�4�=I½��ѽ���x=zew=7> ���<�$�=�3 �?�<��>�Y��IG=�xռ�ct<^��q��;ת�;5��<�yH;�u,���5�O�l��Dj��=�=?dJ=�j��<�=C�=#��=0�P��
��E<�7<^p����=����D��=ۑʺ�,=Fۛ��U3=�����>��B��L���:�A0��8�=��p:E���IPY=�I���ذ=LQ��(ȼ%�C<���<����S>�	>��׼h(�=]�:��*��U�=�yF��`���!>�3Ǽ����fy�S^�=d��;��]=Yi�Y/����><��q<j�`=��#<�T��c�%�<�㍒��F����qU�2��O⽵ �<&�E�c|�=�L��a�=���<�|(={V�='-�g�o��u½�[G=G���R�&=�3�=g�=Kk= �＊�S�#%�� <�(U���=�T>	%�����=C|4�=ۃ��{��UW<[P��W8��4v=�.;T�>�	#�<� �=���<P��=��=v��<�b(��	�<�.��Ђ=�y�=�׆=!���!���*<u���k�=�t"=�4��<2���V����?�r/�=����l;��<,��p��;�ס�S2�=Gm����<�uA��Խ�=��ͽ��<z��7=D�5=�N="�-=T����et=�<����^��R�Z=
�=wU=�e�=�L=�*G����<L����!<��o��mR�Cs�����T@�&��;��(��*żŃ,����F?��Swӹ䂹<�F߻'Kg���<'@ ;�Pڽ�q�=*�=<�2=jB.��?�P'\;ƛ=�'ջ_R<=�^ýT���<6�����s���=�Q ��l���^�<�`󽆆)�[���f8�=�<;��2A=��>��B	=��4��+�=nν3�E=	 >0�<?�L���=�I�RͽaC�<ڕ�~Y/=lJ=��s=�����T�< .��{�;���<���<)��=yW=��.�^K7�����+/!=z������!�=oJ��d�<���=�����$R�0�� Z=Q��=yv-�^"���)�<WF��G=j=��;��<7�u��OC�|Z�@�=-��=�\��pü�]a�:~��`Ԍ;�E�;Yj����;��м�NE=�@<�i���Z�0������Ò�=�[�c�佚T=+lͼ��Ӱ}=@�I<� ����ҽ`i�=�W��0X*��t��䭢�5<��9<6º�|�<K*{=r�H��,���\�=*�������=�^�5�Ž��=~Xݽ@��<�G��k�e��;�=LZ�<�I><X��=<=�.�(�=k 0���XX̽и=�|˽&c,�[ݐ<�fM=!����*������Iԕ=�⽻�%>6�ļ�=�ށ=�	�r$=�������B�=WZ�=J�4�m�b��\1<��D=�o	=�[����ǭM�p�U�ژ�<�`>�e�=�Y��}�t=�o��8�
>�����qv�.�K�M,�=�kj�*��<��m�ϼ0��=��R>�(>�&S=o ��Ƌ�=�3;=I{ڻ�')=v��=����QBO=;z�=���'"<p뵼��=�c`>��k%���?<��h=��A�S��=@�>t?�S�{=.�=�/>>&}>�?}�Ǌ��+���|��r����)U=�[;���=�A���/$�!����r��E�C#>>k���/܌;S��=��S�=<<�=��T=�@�=����]�=�nP�O��x�<&�:�FƼl*=-`=7�����#=c�c�C�=����FD�2hL���<�r�=����'����=^L�=���=��!<�
����u"�zO==R����۽H )������3��3�i:ɈY��`=ξ�<�E�a�:KО��L?=��H=����)ۋ=�m����=}悽y'$=XZ���*��1Jw=��=�
>v_=� >�����¼�JG<�ß����
S�&�$����= ���\ʽ���oS����fĽ�P���d= ���K���(�=ސ��;ܼC��=H�:�;@����<V ۻ�����"����<�����2p`�۪\��y�<GbN���޿R��
ȽZ��A�=�b<���w�q<��s=�n�<w%����<�ю�vV�;�L<�7�<GS�<��=�L�; lܼ��7�jT=��g�0u�:��=�#>�\����=��;<��f���<E�����P==uU=�-�;v�M�/۪�S`c=G\�&���= 6�<a6ؽ�'=��D;�=�=f}C=H8*=���=.���n��=j��!��p5p=Z"��:��=	��܁<9�d��Y=�}ڻ���:�}����G��_<yRϽ�N�<v�l�#����@���=�f��3�=O�_�GE������b�<��;]m=�!=7)�;.�;=�o�<�l���P=��T=jo=�<��Y ����J=��T=%��=��%Ǜ��~�#��|�l7=���"��:�Ϣ������D�;(���H��&^�<���=�'�]*"=Y<7����$<]>��<�����)=o��]����4�M:p7>���=T<}*�����:1ޗ��[�<�ur��(����I���ֽ����<�*>��8�=9�A������vC=;�;�b��=g�����e=�=��6=�p=�k�&\.�dÂ=��޽'H�>���=S������`�� *��%f���h�����Q��`��UC�2��l.J�F�>`�r���\<@!1<;�q���� �i=�k{���</�=3�н�
��7��=�l	����b���_z>��L�㝼����װ�=p�:�d�=AS�< %_�F=�$�=�+�<�ǰ������&�+�t�\K��M=�&�<c:���㣽]|��|���X	>�{<�Ք<�:<���c�����8(=�c�=dҾ=�'����R=ScR�����Yә<�l=������=K��;!9����.=�ha����=K1�~dA;��<����rZ�$�=b�)=�]�<�'��@�=��:��� �m�'���=)��=�<=#Y���D�
��=��uN�<n/j=�|�=3,����->/�����=څ�;�k<�B=��׽L˽<M��Z�r=��>:8�=š�����=Գ>�w�=�K<> �!�Cٹ�a�����G�b����`=�I����b&�ǘ�R����@�p-�=�g�=�;�<@p�<U�V=p��<B-�=�X@>�=��M�=@���ϱy����;�ǐ�U��=.�нds�Su�='�轋\�<m�<<T��l��9��=��3<7�A՝�Е=o�����=�"��>=�17�\^=���n�<[�g��-�=�]
��4���~������<ُ��OR��`"=HԼ;X��<�S=�rc��[�=��*=+ܽ��;٪�=Xy�<K!����"=a��� �=H�N�wlh=YYh=c�=6늼��A���z=a�e������<���=؇�=��Z��6/�7!==@�����=Y������LG��A?>b��<�>=�<A*����>r1>L��쐰��%T=��4=7�<�V�<=t0�=�x�=�X�<~��=	K����I���s��:,����M��k�<��=xЂ�vi��RлE������a�=�<�����=��d=�-�����޸���:5�"AZ�꘯��/H����=�T��=!���o>a�<�n/>�:y����y�C=`��<��9� >	�
>G��=���Yi=f�[�c0�;��V��|��m����x=� �����(�<�8rX=����?Q=�=��4j=�0�;�kW�<�$=�:o=��=��@=7�g���5�X�׽nC=��=I�=�����2ս��e�S��<��=�>�<���Vꋽ#~�=�4��z�'=�Q�=�$��TeC�&=�g�<i��V@�<�n��d]=��=��,<��A<PO=>6y�@�>h��nƽ(v��5.d��y	<P==�áн���=S����3ɽ룽�β<�l�<��< �~<�����<h��ip�	L��_ *�";2�5=_ļ���;Ɖ;mNǽf���R�=Mu������1�=#Q<�,�<��b�1E��^���⽎0+>�L�� ܻ2i��S�g<�?J��v]<��/>"5��;�=�tP�,$ �_�->����:2,=��iM<>����Y�=�Ur�����O��x �����=\{�;{U�ʲ>�������������3��F��_j=/�D�H{��A�=�w�[��R����V�ؤO=&������������=)���Ż���<�K,����}��<0�%=�������hM�D�=IhV=nő=��=Ʃ<�S�=��=A�<3���ݱ��J�ʖ�<G�s���n�?�<��ؽ��(�=���;����Q&��iu=Ȅ5=mH5=sք=*-�W2o<S�=%���&H�����bR�_M=��=�ý·��"��Tɽ�Ž��*=$�����|>�ʓ�e��#��8&�}���XD�󖑽�x�'�V=T/=�~�==߽�?�����=g�!<psa�D�w=*�=�!������=WMX��>��=OR_=�|>�V�[�1>�.��4�-��h7> �ｏ�0=�m�S�S=������
>�5�<�Yҽ#u
<��=���=�/�=�8���c��t���W�~>!���';�3�8�M=*�޼����vŽ3|��ԅ
=;�<E	�e�{�'4������p=I�=��>`�\=�.x�h���K��;i�;���T�Ͻ�fQ����=�,/�Eǰ=V�<���q=i=ü�/(>�y=*�+=%Β��t?=�̼���<�>��=d5=�����O`<��pū=6it<H��=��=����h΅�����g�=OR�����=0��<&=����.�=�=�=�=�w�/��F�н�D=�A��=+5��{�<��=$�{=e��O�%>�70=�H�=��ֽH޺�0}�M�~��=M8�=� ���&>�]�mF�:�x����~��Cg��p;31=FN>�+����;(TP=�����W;ɠw=�I�<^�=��<��}��4*�l�:s#'=W�3<	 �=�<��	ݓ��&h�THk;�:㜊�a�
��e���<w�����=����	 L<9v��M�{�Q;�=�e3=Y�,�v/��J�m=�e��
�Ǽw�����y=O5���s�=uU����=v�Ϻ�0=6�ܽ?�==�;0�k0����<�Ez<"0o=?����$;�Ih�2_1=�轞과�슽�7m�E3L�v0�=q�1>>>=>�f=��%=5Ǔ;.�V=yr>��>a})=���<���}s;����r��<��>_`=�><�,�=���=7���w�=�䑽bP�=@v�#!�N��2�½r�;�G>a^��ҽ$/�=��	����=���W����A<<C�:����O����<NG[=��:�e-�*�;��<���׽�f���8�=��罿���F6ƽf�<Q�=~��� -�>��9��tƼ��f=�d=]l�=:�7= ����p���
�=Lu"=���`K���ۼ�t�=���=�C�=�?�<�?n�];���,���~2/=�ݣ=��ݼƢ��0W=�a��ޙ=�p����=@>�=M�B����<�D��M��=t�d����=��=Ѹ���ܘ=nA�=��p>��=�Z(���*=m���)���	��5to=(k/��[�������J��~��@���q�<@k�<�?v<iq�=Gz=HzH��O��.�"<栭�7�
�����Ͻ��;�<�Cu��E=�N��H27=
�(�ٵǽc��=������g�лH!Ⱥ�3i��x�<��T���<������]�!�=�H�=yLR=�n9�Tv��=5=\1=�u>���tzG���@=ø�"4�ƌ>y�{<Xb�����1p=���0�.=����H>Ͽ9Y�D<;j���(:�8l?�x�=��ʼS��:���=�fl���J�Ǔj�}��:O���m޼�� �u���o<'�=����ڢ�����;���h	=F� <E�<&aȼ�Q���O8;_�	<g�9=��'���O<5�6<�~���S�,�t�������< ��<Y�>���%� aD;�P���u��'�ݼ�u�������<�쎽p����S��<�=�<E���,�D��������=0�=�V`�5��<|<���:��=�6^����iν����J�<�*���򽽙�v��p(�H����=f=�tc<���=����A;=�!<Z0�� v=���<!P�=aU�<�U;_>�������!��=��z����( 6=��L�O����k=t��X�:OQ�=��=�%��޽��|
>�=��=q�J�4�����;>W��<+�Խ@.� �R=~ڼ��A�լݼ��;��[�A�����v=�k��u<�6<��<�d�9�Q�Ma^=�E/��gc=B$�<��L=H�\�
��=
9�=N�Z<%�8���<��q�~���ȏ���/=���=��=qgj��8�=�9ý��`=����RӠ<nu;����=��ů�P^ֽKJ�v-�����;:���j�>�<	��;~�J�f�;˹�:ℕ�#��=��>`�<q�=�X�O���1�<�ܙ=*����潅�ƼbV�=��X�e��<�F
>w7�������<�~]�r��<��ռ��;=	��;:T��y�=��;��V�=8��� ~�#4=,.={y�<[��L���= C�O��Y�<�M���|�0��V�=<��<�G�n�>4@��YD�=�&��3A��@	��&d=���<���k^d<���������-;�N�<�;��O��k�=�OӼ-d�4ǼJ��������)��� ��c�=#G�ϊ�m�N=�鷽$y= /�;�a���:a6Y�N�=��={�J�U=F�n=�J=���<o��5 D�i)=.�}=>�=��)�<R7�;�<Ohm����=c>�=P��=�k=Yσ�D�[���=���8r�r`�=�u�=�� ��O�=*�<���<�2=�����i�<v��T�)�H=�=��<���ED��N�=�V�=�;�p���\�<�=8�h=�X���f�=��I�f���5�0���M�_CR��!B<�:�=���Ķ��\ ;�*~;��"��%����*�=���7���A�=�G��Fy�=�=�<"L`���7�GC�� =N\���_<�}�>/Ǽh��<t�<�6=1����=P�X~���{^=3L3�� ��7���\<8�Z�/�Ѽ�I�]����e�f��<�5��;�Dڼ�7�<_�ڽh�I��U�H)��D�f=�rK�'��!��rw����;J��=�ꤼ��>>Ґ<���c)��r�:�
5�=�+;��.=���e�< �ս�|����b��=`/<tz=����D�=�>���l�7O��D�����=5%!=�H�=�,/=�=�3ʽ�ߡ��1Ļ5%�<��h=��=ᄘ������=.A�<���<�T������=꨽�9��
��U = �A�k9���=� .������>�M;̐.���<��ܽ�j
��R�=CVU������M���<�0ټ��<���<H
>���S�=�k޽?41��=�q=D��<��!��J��J��~|=����>S�<�j=��=�*.̽5]=I�"����=W�ռ��ѻ����Kr���=��B�<�默-��I�roc;���=����e=��J��&���r߽cz�<���=�w�8�=�E�=�Tʽ��=(�E��3�=�4;��8=�+<�N�=)�5; ]=�1C�ھ��]�=S�<=L�=���=����=0챻��4ģ=7O�<��=�wc�K��3�����<[O4=�A��˷<f
U</���f=�K���������Ļ<PbN=u����!a�Z��v��µ��
�<�ۼ5
�=�a��.h��@Ó������6�<�r9�s���$<K�>=B��<����� >B�<Y?�����=��<�v�q�����=��=�>�;I�ƽk��=�l����?4>��=3X5��>�<�=�E-<��=�F�GA��z�d=�\E=����3� �"n+�}�
=H4<��<�{=�Z�Q���<[�X�,\=z�	�-%=[IR����<���<���=�ς�8Oٻ�� �>�l�	��<�����]��~�<6�G=�˯=9	��{��=��t�fS=f5�;�D���D�;Sf�:;Jy��8)=�����G����B�������<��=���=���\�=z����n���9W X<W����K=�e>��ټ7�'�g�>��<=Ƹ�;��>�	>�0<�<{���]�gP�c���A>����>��=Z]Z>$�v��tw=�� ��Jb��Y�K"����6�G�i<I�
e�<=�]���o�7�=2����>SI�<N����;a��p���^[�QH�=��>C����=B��<a�b�*�O��.\�7C>�A������|����3��r��=�=�������<���n��<9�;\�=Q��˿ͼ�L���">\<���p�V�E��(ǽ�9� =����co���&�=�5<D�3�D���<���<��e=�ه�+aj�dI����q��ͻ=�A�Y<��#�;�f���&�����y<7�>��j��d<�����2���=��+�\n�<!�a=~�$���$=.,+�2l�=��`=�p(�Mp�)o=���L;=?[V;�=3F���')�C�=iv	��Z<p�<6��Yެ=|E��=<ɾ�$O��z�&�� J=V��;̰^=�5���3��_�=]�<z:��ԛ=^
���铽���;b�'��ۼ�!��<�)��))�=��<��=�b��k�0=Y[��F=�4P=� ��PN���z�	�=K6���=oG�;n��U�<|g�<�����7=En��FF7=~ò<�x����<�o+=��O=��=x���N�����:����1��ԉѽ�\<%��=�Z=A�����=�C�=��m���>��:��Ѽ����;~<��I�/�F=޲<p�<g�'�	��8E;6(���Q�?�]=h�'=��>O��QH���L�{;�ȸ=b�d=���=��=����&�>�0=Â�=K�#��<;>�_>c��<��(��?%��"�=`:4=��=@[�=��=���=~Yr>��b=V�K>Ǚռ�3ܺ�4��I��3�<v5z�Va����$=�_�����v#�<� �<�M��b�rBM=9��=����Y����y����ཎ]�=\���꽭׉=(�<O�F�a�a�l�]���P�Ll�	e^=c\�H@9<����M
=��<�E�����e=C�;=;����*�U��*=H�� �=Y�<_���&˽\�D��>�]&�����_=SHz�@�ͼ#J�=�"=���<��U<�Cнnč>n�ѽR��<6�?���C>�����#	>B�����?A"��=���{��<��@��>��0�աd=a��<�2=��=� >���<�p�<l�]<~os���Ӽ�J��f�"��� ��=Q���;�<F#=d-N�E�Z�ey�=���<��@�߆�����=��:<fޝ��ȧ=���=���=���<^�!.���R=�޼�`��"��c\���l����eS=�
�<?���A���.A��7=9�7�=���=+A=�I��=:��$�� ��:s��sf�mU��}��=
$~:�`P���K=�U=�P0�ߕ2=��s�����X�=Y��<V&T���bn���м`u�=.M����y�.�Uc>.i�=��=Q�޽��>"�?==	�^�9>�̪=����(L�=�;/��N��b�������D}>�ؙ=�7�=�;����=�D����=�n=�ؔ���0�'�׼� .=B$	�B�}<D��=Z���Y*��=�:��F���G>,ۻx���C.<F��<&��W�Ͻ��=Ȉz==-�3p�Q^����4=�+Q�ݞ���<<��4=Q\0�ם�<��P���;ϕ�=��>�G>?<^��#>xHc=�m��³że�=9�<rL>4�ֺp<�XH<� �pa��P�<��<�@��ޭw=��<�x=���<��=�=��>��Ѽ��*<z�G�rĽ��=VdV<ۻ
�%�p�etP��~X����;Ύ=����A�^�$l˼m,��8��~�=j��;p�_�Gkļ�d<�Ѽ�7���\A=>#]����=u䩻���<O�$�$U"�W@���D4>����R���=�\�;`�=�]�<���<=p=1G��c��;��3��'������܋��a�E!+=
5K�"�p;�+%��&u��x[=�(����F=�4�!��=UT+�Om���t��Sۤ��=��;D1ü����F=��켑��=C]��!r�@a:�LV�=cC#�(yD���޼:��<�S�<�Q)=��P<
:��w�= �E��$�<M>�<�N?<88�<w@�<�@�=k]=������K^=FFp�t8�<��걔<�%��TIB=ri'=���<�-���ap��u������V�'�5���.G�="����C�=cɜ=�sA���p=ǌi����"��Q콬�)�JUϻ\i�=�,���n=(��=�U�=��=�.�=%=Y�= �,�����!=O�=]����=k8=�Ta=׸�=��=3���J=��7=O�i===�G�����:�H�;K��<�ѽ�Z���缇�=�-�N�==��<�i,=��$=���<r��O<
D��%���r�����;�Z<%#>�k�:�>�T� ���^�<.:���,�<���=��ӣ�=R��<�(~��73����`���m>�U>��=�|�<�k=yc�<ߕ�=c�=��Z=B9*=J�=Eq�<]c�=�t�K�=(5��N
>]S�=RN��W<�}��q��!a<<�p�=����b>��7�=I��=�+=Cz+>�,�ֈ�<LВ��P�=.�P��,+=��:�꟧<������˽;S)=��;���gȋ�-P�������&�R^l;��=MW���^�]5M<v����{=d3�H��=��c=�3�;d=й;�7�5=Kh�s�����<6�Ὃ-4<9�< ����ۼx��<'��<7��=u��;��弐r��н��<�"�=.#�<�̼{�>;B� <߶=��K;}�,�?��
@���i�ض~:ڗ��&�c=����y<�'Z���=��w�=d���"�=j�X=Kh�����=�����=�]�<��s=���<�lk�����i�<���=��̼ܣb����z|�<C�_��c����<\1���SN���=銺��o�[l]<�
g<�x<�N����==�=���'ݴ��%ļo�<U#�=U<�=�A�<iR��z��P���O=h��;�ʃ�f��=D�C�'餼�=��v���<��޼��x��jU=  ����=�7��>�<�����A;�	q=�3Ҽ� �<z����_X�\��<�&���<*������E�=J��<�@�=	2S=�v�=����s���� <s�=q˼��i�5��=�x;�/�=-�=����"=Ҁ��ݽ�1G=R�U=	�=9�e�gM=��=�3��a<<(��=�>!Q;QU��GR��<sg=�z��K=#l�U�<�\�D���l��=�p)=��<}W|�:G��_K]=��Z=e�9+Z���b�<. �;VÀ;���=�C=.���>��4��M�=�f���QA=��x�<Z��<�=A=�L�=��=�h绪��7�#=|�R��߃�~*@=)t�<��9�W���%pɽ��r�V"�>�=:_�����;�u=�L8>�B`�lB=�c����=>��<\׽w$V>�=X7��y!��{%�Jְ����/3Ƚ���=J�%;��B=��;]L���z=(������=L���g+q��*>�n�	�7m��/k�<*QN���<^�:�4�l~��������=�0ν�' ��dI�򋆽%�=f>�
ý�V6��?�]�>�����c�O���i=��<�%�nS��Q�+<�䜽d�>ؽ̼q�=PZ<�5=��
�8P=�ڼ��/<�;�����';�C��Ɏ���g>��o=�"�=�ϼ��u�u�\�RAϽ	�_=u=z=���=(e��g�=���=憠�d�=��p��[:=��D>�;�:)�<��S�S��=v�޼X�=�x�<�q����<�<��=xc�= QJ>�(&���o=�x����)���j��<0��!6�<nmF�!��EN�a����$=`��=D�n=Wg��-.�<�U�=a�1<_���@�����<�u���"�<� �Z��;���=O�;tؔ���=7w�Gm!�;���_�<?��<��=L
�=��\��BJ�)(�=^���F=~�=�VS=��`�&��<x �=��󽣥�=O�f��!=,��:lV9</6=}P�;��[=?����=$x�;��C�۳�<��=�%�=h�3�����UL<#tL�a�=���t�<b�X���L��e�=��L�g����V=OU.=,��=^�P��M�=�@��A6���`=oi�:	r<���1������1��3@߼��A�����<��>$R��k��(��=^=D�q�>w}.<��k<���7�x����<c���x듽b<->r~u����=@�����c=��T�̻�Qi=�v��1~��c�޼� =u�=�I�͹=F	�S����<��r�~�=��<G-��G��{U�����9������=SO>��������i׼�H�1�k�7=���=r ὅ�j�c{A�K�;����n~
�����9��G��<��x��M�$�=0��<lH
�~�;��J=���;��(�[�j<ds������W�]=���=�{���!=��M=4�*�&��=��=4�>D�=R� ��b����@�< h.=�E�=<���s��qڼ�A���ӽ�C�=��<3��߼�������*���L�<�����=�źp�+=BG��擈�A����%=�3�<^ƥ=9�=�ċ�w�@�;L��*��Lǽ�$���hH=�:*ś>��<����o8�#�T_���=�	=�	=�x�Ō��H�Y=_ ��a =��>v&�=FM�=OM����%��<�d�;d�==;Aֻh�=�z̹����?v=���x~=���=�,>�}�=S߹��Z=}A,�>>�=��P=�ʪ;��<a�n�a�{=��<Q5�=�b>�^:��B==���"=۽��Z��$$=�v��7=��<��[jż�;��,��Pw=|�A�OH��c»��[=%��<�y>9ɜ=V�c��ey�Cy�=%d=��Ϸ��0��;J޽�<�pr=b�����<��O�1�Ὁ�(��,=cĽe�>���<��=�b��N��!�df�<b�>=��<���]/½r��>��<ʋ3=�}��>'����=���xҌ<2kn=���=8O��<��\J��	���̻U̍������S��\d��	="Gν�&m��yӽu�O=�U�<���=���=���=iᕽA$�=z�Z�9V��3���A�ڟ�=�ڒ=!����>���=�ׅ=��ٻb�r�0c��b�<�x����X=x�h=�Yн�oa=��[���G��=�Ul=�
3��#j=Y,�)��ܽz͠�h�>]�T<���=Q�.��<��kI��2�2�x���z�r�~��&��"��f=�b�=Vr&�WC�	�S=n��.mL����<������?=E�=GH��/����l=ߡԼ������"�.W=3��=�N"���j<�3�< G�=!^��i�<�μr_v�v��<�Hy��21���&= f������|�=�=I��+������Y.>8���ڽ�Q@<֟�r�<�1<!S潁�;��<��<Z�8=<�p�4>�G=��ϼL�� (<�v�]!���d�<{g��)��?j���߽e�/�<�ba=�����w����=}|��xU�=�ν@#N��$�<���+�4<qx���(8=&`��_�N<������}��=jX�����;�=eFͽ�0��i=�ݽ��=$�=�����"�kQ�<�i�=�,��KP�nj�<���sxK��M>L����b=J�=�?|���r�����C����=:�
=�A<��4�<=�H�v��=P�˼'~���OZ�1B"�=w��=����o��O�Z���@����<���>�^:�~X����=����'@���6���ث���w=�;4�Z`���b/<�X����-=z;�<��=�����%=�`�3�>[��<�K��M��]���;$»����t(�*�P��v��x��~�<lN4=3���`3f=/6="1��A�[�E�v�w~�<P<��=QH[=p���z�X���<y�w=h��<�n3<f�������3���d�E`(;��߼�iX=�9�=������#=O!H=������B���=��><r%��H=n��@݇�vr=*�����S��
�<x�<�A!����:6�ƽ��=��m����<f��ɡ�<aoɼ+���?��=�={��=.��W%+=W�	=̻�=�d�=�uu����<��ڻq�S=܁ ���T�� �<����޽���<���<!3��0�[.�����9=Z�&=o�o�*�=���=��0��W<S�/��6=�a�<� <�VĻMud=�押���<V�=������Z�jݒ;�/�=B5��O�X��<�=>0���L�Y��=���`q1<u�[<���=?�!=������=aG�ZfP����.~��j�=�Ҕ<��=E��;@�=��,=�/|��<6����/�����<O���z��e�;�]<����	b�����/�����9,	>��2�<�N'>��}<c����Ɏ>�=�=�X=L�<乽�֧��Zǽ��h���=�7�=��>�P�%>Wc�H��;%����.��D!���=KZ������c��<�I>����h�>�!���=$�ټJ�i%�>B��V�u����(>��>��-���������0e=}"���d��;���x�W똾��%='d	��L���|<�+=g��<�m�<�r%���=�v$=	 =HY��(�����,�Y">�T��EL�1��K˼3��<��0=�h�RV���=�Օ�n�"��ę;�p�_�ػ��r=.yL�P[3=��-=�P�
:����d�g��<8��<���<�=�=�ꋼ��˼�!r��ü��m�z�=[��<j�<���=��{=m�=y���65��J�a��ލ�Gb?<�<��_�YC=�}�=�_ͼ5K��B��Z�;.^����K��="3=�>D\=��*<��ļ�����)��fн~����=3p<�:�<ז}��[���(;ut<F]�fH�FN�=���<�GT�ń1=
{+=��>@p�=�T/<�(4�7���yG�˽��$=�O�2{x����`�<>�=<Z��<#N�<�]��x��Jr������	>���+>���� >��3=i��=��ݽ��=5��=�=��=Ț��o�,"!=3�<jn(��ٽ�eV����=�
=�g�=.�=�n�=���=�%� �\:c�
�?�����=ϨF=��1��:=z�=���9m�
J5�(H9=k�><�n)=/d=9�=ɲ�<S�+�	ڥ��D�=����઻>Ca<ƞ�<Z����e��x�=C!�8l=�����7e�����~>P=�=�伞]<�=U�	1	���y���=�&=���=þQ=mn�=X<��ȕ��L��9�iۺ4k޼f}�=����=�=�1��$/>ȗż�+���Y���C�ow�<��&�{��<�����)=�__=̰��%�i�a��J�F.H��<O��<��-�P��l�K=@7L<��X=1��6���y��<� � u�m(�=@�f�'�'���|�P�[����;��h=>c�:%�y�#nX��ý�r�od����=�s����q��=D�F<��鹞z�:�s�<J��Ƿ��غ-09<f\%=��Ƽw��=!��;��C��0�Y+=�ms�������=�͈�Q�=�<���=c��= ��=-�=R��=�"𽙡"=� �����i<Ǣ�<�8=�Լ� ڼ�5^=0�p<�h=�	=�m�<A㲼ފW��g{�Ӛ=Ic�:XӞ=3�1=�R;8tg=��<�*d=����=3��=�6,=*��;������:���<��}�����m�����=;Ӕ��żR�&>��=����Iӱ:졈=����x�=Pi��錼��#��Ӭ�<0��&������ۿ=��l���C=(���jM�;��Ӽ�u��z�<|Z=4;�<���=;˼u�x=�큽N�=��M���P�}sw�n��;¡�:���OU�H�����*<ꐀ����<	�}<�^�<⦓=�UT���廉?4=<e�=�#<��;�
���$ѽj>=�<�=��>����6�<j�Ƚ� 1����#�)�b��	�G9�m��ǖ�=�<f�~<�5&=�H�=y��=l�U=��x<4c���=�W-��S�� �l���U�<{�.=��)�o�=5$=�V;ݑq=H$�=�M��(7��.%=Ѓ�=k�9=��=�����$��=�;��}���H=o$�󾫼�"�<�<�$o����B�=��<٥�=���<�\<NDż��:=$�������g�<�ڽ���f=�� ��(
=�ρ="i<�{���$G<�f2����A[��WQ��t�8=���;[��=�cw=k�]�'��;G�u<�|n��hN=^�$������ۼ?�&�h�r�S7��$1=��;%�K=�<�-G��m��<S�<��\��+y��ar�*[��S�<T:��+ؼRXU�n�ǽ��E<�9>��=z^x��r��&0���	������߽��7=ò��3
��A���R=hj<[T�,�1;K.�<E@S��L=��:��?�<ڋU=ꃫ���>�#�=�{r<_o��N����oͼf��=Z3�=���;�ɼ�4��LGE<��=�;:��=bs�4��<�>�خ=�K��i��UE���B;���n��=Cw�����=��M�Ϙ�<��M����u񼑪c�)��(4+=�nܼ0�d�µ�:"��=�tm��}W����<B\J�&��Z�e=�O`�H��<N�=��d��;���%�-=�#D���9�}�5��܂<��3���⬡<m=��S;��Y�eƮ����R���o�;�Ť�6<�����ֽ7x���f��5�����&=I>
a�h��B�a���g=��(=��<�i�2�׼��I=�4�������=�}�=� ⻕�9=��x<ip��L�=4z0=�p4��&�{�=Sb=�-��k�0<�4�9
��=c[��x,�!�:��(�;��5����=/�;����T?=[��������4G�2p���u+��1�=��=�V�<�sH����=*��=��.��<�7����㽀+����=��<��	�+��-���wR�A��<�MŽ��I=������A��"��� ��\�=E�μj9����8󪏽vd\=�Xo�?�z=P�Dn=�\n��<��@��U<������=��O������FTH=���=��<~�=<�2�B�QǼ�� ��ʽ��Ƚ��	>��m��W6��F��� �ID�=�����=>����e>����i�����Ϛ���_�<�ڇ=.�i�D�9=���->Y�c��>J�A;�τ��\�=�f�=M�=�}�4��=B�����̽5���\��!�<�� >���=�ֳ=�J<
1G<�� ;\����|�=c;ѽ~�S��f�ZwD�K��@=Ƒ���`N�f�м.�=�=�J0�t?�=�\D��nf<z9C(�<}�0= 6�<Pg���=�tS�;Y���30��
x���B��������Ȉ="4=�A���=�M̽|X�<Q4U<����9=�'�=�/4��g�<�r0���>8� ;���=�j0=P:=�۸�i�8��i:n�w<"��=f�=�Op���>S��:h;C�v���M���<g=RG���``�=
�9��/̼^_�=R��	�;�����.K>X����U7=�������;�Hs���M�p����q�}�<;���=�v�l�\��~���Q� ��<��4=�q�����=a�8=F��=)�;�����N=LM�Dn1<�`�;�4�=�m�(��="u����N<���`(=G?�<�X���^<�,�=G#�=]��<=�k<����څ;�"��d�=I�;��<A�]��Xh�z��=�y�<=�=�=&�e� =�.e<��%�Vq0�9���:$�㌥<�t��0�Y94���s=?0=�y*>�V��K�=��<< CI�+�X<PX-�T���Y!��b<2<���/=������� �<�h��gTC�戴<V0�=��!=$z��aj<il�5��=HU��P�=�־��<x�νǡY�A����uE7=g$��W����d�#����<��y<�==Z�0�T9>���W�y,v<1ȵ�9DK=��<��0�Kh�D��<�(����<�1<㭴<��g<~(�fE��d%��$���<�$.��/@;�磽F�=���g�<9���ۖ{�6=���<���=�dP�4JK<a��< ܙ=j���o��]'�<��T��^�����<�ؽ�C�2q��|���ؼK&���^��I~���a�=	=�Aۼе�Q�=�a����=Q0��F����)��I=��=�y$=��0=�og�an)=uᔽg룻G��<��<��$=>"��Њ=�^L<5�=5�=4����(
=��ٽ�؏������=��5�ʊ�=��<�B=��1׆=�A���>�=�����<�U6=�Xc=��<���t�=�=�[�=�{�=W6���� �ZFҼ�D_=R�^�;p�=��;�N���5�	��<5=�<��"���]>��V=�Q=��2=��lP �G]�=laW�Z����@�=׸=�����*i�٧�=�C��� �Ҏ���5�!
2�%�u<��"=�Tż�2���k���1�=�e��(���ݼ�v��� �=����N���f>�R�u�{{�<c��<��&���H=�Q��3��ֶ���/���5j�=��e����m8s=]�	�'z�<7�r</|d������?�$>�$<����n�=�������<H�ȼ[��<tQ���G+>X�ͼ�Y�=mv��G�����=Pi޽u�y�щ��j'�� ><���=X��;3��=��=�ib=��P= �}@���f�=����9�X=�)�=�����#k=��<�=���=x
<u�l���C�9T=�!�ڗq��S��[�'��P����5*>���&��o=�}>yx=��<���=��o=\I�<��x=��<��>��=��/��p��# = �<�LA�[��=�b9I#н󧽠t�<M�<�l�<Am<
v�=&�x=Q�_� '�>񃺸�<�,=�iz;t�����R� k���Tf�P��<��<��<a�Q��樽�׼�v�=~h�=��"�<���뱼<$�<'Y�<��=K�w<tZ���y=,A@<ք��i�I=*ν�ߑ�������8<���=d�r�뀽03<.�=��f=5(1<;D���j�<T�k=T�>06$=�G��V��;�-�{���Ck��,={�#��Hi����ظ=��������=�*ҼƓN=5�S��B�!��=C���WJ�=��
=y�=�j2<����F�b=AX�=�͎=uTW=(�=I�b>ݺm�ڃ?=�t���ҽ���y�Ƚ䁣�E5� ��Ď��@�\<7�Ļx���E�<� ��`:н��	�%̈́=�m	���P���e�|�y,<nӿ�����;=@��;꿵��2����
�D�ݝ�=S�2=���=Ꜣ����p�������=O��=�$ =����g���6�J�I{x�-J6=��\��:�=��<�eR<_ﻫ~s�,_<c�0=��2��ޯ=�Y�= �">�ej= ���@�<:��lЈ�ᯢ<�(�=���=��=^��=t����C-=!I�;.3����)=�߇�
�P= �"=��x=R�W=�׋��������<Pٗ��C����6�!�g�O0>ݗj=��k;?�0=�6�=��=W� �VLm��J5��(������#=�i=�v���0���n<�S��}!�&<�;tnǽ��0�J$C��j�=�fн�T��qe=�F�������3)<	O�Ҟ�=�
�P ��D��7̽���=֛�xL�ib	��/6���t�/2�X���b���S�Բ��u�D�}T>�%̽E"�������7J��v��%�<� !=l���Ğ=߽�=�r>�q��q�=�x��:�m���.�Z���6�K<��W���W<#�=��<Y�w=�Ε=�|�;���x�ּw}�?y��9�+Õ=_g.��V��	�M<�O����x�-Up<���=��;�(�<����*�6??�����L=~&4�hMH�t�=��<9%�8��;����$bB=�]q������F_5��\��\5/�A����r&�������Z�^=b+�<��=S��=m�G=�rN=��=�O��>��7�=5�4������7��*���옼�9�V�=�]b=�{<Ԟ�<n�;�&E�<��U<����<`�=�!��������g�DP=��t�R��=�>���"�ؒr�j��L�Rc��30�=0Y����>�/��b�*����]<�B4>�/��+����<�,�4=>�ֵ=�TZ=��M��=�>8ɽ��y�L��B���e�=9�)�lU����"��*L�	1>���=G�=>A��<*6�=������Z�v�=�.1�~�!�����P8=�+���>�9( ����M�cV<ǉ=�&�<e��=S�=p����\�=�cQ�\%���G}<'����<�y���j=D��`��a#����-<�3�<��ɧ�<nDS��}��s�O��!=
M?=�Y�=�h=D5����<{e���K;��~='��=\�Q;�%=���=M�.�Ҹ�;޳���A�=>�h����<$�<�ྻ�I�<��K=s�=�>x�׼�r;=zz�Q
k�y6=��O<�t6=�,�{��a���s����a�(�,< *s�=ڇ=�%.=<�='އ�}��<:��=֤ =b��<.�h=vT<WCI=)&V��WQ���=�U=,>5=���<W�D�,����⯼rn�=�ݻ2���<>����=Wk=Q���X!d=�*�-���N�F=�m	���Q=��:�N�b=�*����=�Ŝ<�׷����Q�=��=����"��;=�1��*�<�
�=9�=���tמ=������ =���<������4=�ͽ����8;�g��]�;mμ"0<���&��FK�M!�=eq�|�=-��<D�r=j�F>��^��><=>A�����<��XEV���'�L>�z{9/�7=E���=>hp�<L�{=�J�� <�4��/�#=�#�x�x;'��<�^�;��I(>H�/�v��<R�Z=6��=���=b��=;��=��=5gH=��;Ǔ۽�S�:o��>�@�= h�=�t��R�Mr1�P;A<�n�*:Ļ�Y;i?�=c"<(�O�I�b�9!��~]�;�+>n�=bo�^��<�^>V?�z���aS2<qb�<�\��<�pǼ�Z�I���Ni>�\����<����%�����= w=f{�<�8�h��=�=~Y6=nֽ|!$�韵=�N�<����7�v O�] �=�>��=iR��<1B��?�=��ؼMd��1<� � >/.�=��C�L:���W=Ǐ����>H(L:ZM>K�=l켵�<'����Q=Z�&��~�=Y�V�o�u��I�3.Ƚ������=O����>�J=?��K�=j�=V�<`&=r�:=v?=i9�=�s�=圔<�9�"���Ƚ���=|[�b����t��Z�.m=D=�=���<����Iz�=�=E�<���l=�O
���`�<>,�=�<��;��u��=hV=7\�<���P��)=�� O=���B�C���=;����A��,H��P�<�n��'5�5L�=��	�9�G=��&��+=;׽�3�<�;»�A�=05�;oP�=�;=p_==$H˽��Q���!�+s ��л�U=@��y���M:i��<��<Bc��QA��jԫ�<`E��;��;�=*=ּ᠑��=�<`�-=������<�y�<r�P<&�==���7��<u�=�"���A�=��r<H�=X>�=�m�=��=�Ϗ�r3�����=wj�=��x=�-й���(db<����6��=&=�yL<�$)�(I=�=�5#=����7Do��݆�Ha� �8�׼f�n�<(>�딝�PD=�ͽ?:�=6��:ӝ=����=����p=M�w\��1�>"E� Z�� Df=h�˼y�><�1��SP�H{/�%����2��p�=��=T����?�Jn�=�l�=�B� �=HBӽO�Q=���g����=�!=7�f=�ý��=׆��9��=|�D=�	��s@�mJ���"=��n=��=��l;B���p�I;�+M;;�<o\ �U�</�0�#�<�O����;�:<�=3OL=�U��	��<��=ڝ<aP�;�O����<u��=�;l��E�v9�;̓;�=:^��y�0��(�^V廃�$��?=s䞻.�ͼ�[��Tu�;H�=z�<�ۘ���<���<�=3���׊=��K�b]޼f#�;rL�=.���qR�=�M��C�=_7�����?{=���=xHb=s��d�=��=�u��oG�;ۇ=
k�<�hнK�$?�=W&���i= =T�$�Co��*�=�=t93��uغCZ�O,�=L�$=�T=Tָ=?N�=ĥ�;�D=�˅�%~�=�<������((�=�w1��A_=�_�<�k<w�	=Zuq;rō=��A�\��=��#��sA���=<ü��潒��<H�K�"(=���h�����=^<��B�����<�<��y�K{M=�p������/�=�:��~�<y��<�I�<�:=��=ڎ�<f��<�̥=���<!W=�R���=O��=�˽8�;�!��@�=኱=jg�=ѿ�<B!���u=f3�<�X�=UlȺ�@:���=6�>/Z�n��Z�t=%�=��=���������@:��=jm�������p��%�=='=�:	=�E�=�K<��<,N=�ݦ=�Ӓ=�tv=Ң��*�~�)\2<8�=�MW���G=�م�
#�<�qv=A�����|kR;��=��.��ߘ�����,༸�˺��<��=(h=u���_�=v���&�Έ��4��=����x�>�M��=��)�$�=��1=@�=��μ���:9g=�-<���<�ӆ<;��af���>U=H����J���8��l��/B�=��=Lx�;�㿺fc=hKX�<} ��}<�@�=&[D=�$������|y;v��<;x��1�o=�$;�a������8�=�O:�H׈=�T�T)#�l��<�䟺��׺4X��=b���;~������aXu�}�^�ՙ5��G]=7��U	��]:=3�]=���=1�=�1N=ے=y�=v���t���qn=�<�f����=<<��+l_=a#U=j+��y`��C����=�s����=�:�=��<�|���X:ׂ=�����A�;�uO��H���]<�.'��R���f����9=��={�=%��=�\X�C��=L��=���
�u=x����3�"����<�\�=�iB;G��w(�=VK��,򭽃-�O$<o������� T���׻���)== �<��:�����R�<�������=�컽Q��;RP�йN=�=0g�=�k�=�*	>�ޅ=ӱ=~\ʼr:�<kM ��c= �MJG�&=��:�tF����=&T���?H=i����<�h�=����:)�=$�9=�m�?��Ta�=.�,�9�=㮺;r��<�c=��v�7n���=�\����=>�c����5?�.��=ü���<�*<�ļ���8j�P�|�i�K�}�׼Τ���Kϻi�W����=�����M��\��;�P�<�s�=(xF=�ԯ�"i�<#�G���:b�<�,��ŕ�L-�枊:�^o��R>��c<Γw=�t�<2�����"��孎=Q%�:4=��ɼ������=wP!=\P��� �������?3�����
<��?�Y/׼��m���=�� ��GU=�2>�K=�����p���}S=�D <*(���=g�Q��=�	��=b=��
=�1	>
=�4�<)�=���SI��I<��V<y_ =	mx��&5=��=?�l<��<���,���i�� b�;f�˼h�~:M��=�1ս
�0=,�A�#�J������W�=1�c=���=Co6>C�z��[=��=S������)�ֽ
U��
�'��|r>�7k=?�RɌ�Pϝ��4>���=-�=�<=+X=��1<��<�XB�<��=5��<$�s#�<׋�=j?G���(ʎ=�X@=P�&=�[��nJJ<�F=���;[
�=`+�=0!�;Fi����)>a->�(S>�d=��W��}=t���Ǽ�>$�)H����o�J�2>���Ѣ���+v�l�w�X5p�I� >��C�r�f�k�9=$n=L{�<g�w=q��=�%�=��/��9=q�=m�U�Fػ��u=��RrĽ�03;���,�8��	=\�)��+�=���<�\�Ґ<[��Y�<��=�{�=��l<o�R=�C����8�@1A� ��M��<�I�=mB:�s����D(�qh�<��=.��=�ç�=��=h�a����<>��<ٴ�;/]��'�<.+ȼ+�=�a+=�p��ϼ����)��b齾�=�g�=��=TRݼ�0=�ע�ob��?��eT=6.����<����;H�V��I=r�����<�߼�fe��l�w�G<4��Ȗ-��l>��Ҽ��<1��<�6��.<�H޽�O	�R߽�I������յ<Ћ��5�z=LA=��$<�&}�rǞ���m�M�O�#��l���A,��1<�<�<:#S<u����O����8=���<h�<�Y��)h�=�ȼp�#=���=|�>hE���=�<\kq<@w�<�
�;2l^=u;�<�W\�>����n�<�/$=F={�*=Vm�;��ʽ�E�:Ĉ(����;R9�B
�<�&@;�.=���=24*�wp�����=�ѻJ�>����5Q[=�?��I+�=���L��?i�����OX�;Y�<P�i���ʼ�	><(�">B'�=���Kn��e�9�7�8=�[Z=�B����&�'�=�O=ژ'�����0�=�Р��d�=��<��i�J�<>u�=���;Z���j>�E
�C|<6�{�8S���/������ ���B<����X><��=e&<Z�=4�!����F������3߻F��<��;���<�آ�� �=:,�A`��B�ּ�dѻS'ټ� ڽ��½t�=�߽u���<�X�@9N`8�
���җ��d�<��=��<��y��ҽ�n��5�}���ؐ�F7e<�Eڽ��dGK=��ӽ��z=G˻Ŕ�=�t��@E�����;�d�`�g�F�=�+���׽н��=���-R��J�����=�귽��=�ً������=�r���	�S
�=��;=^�=3�m�����濼}�4�q.�v�<2�=�^
��Z<^:5=0K����4��nټU�s=� 1<��=yI�<�_��;T=��B���5��R>,���=9�=mK��F�t=����e�<
�_��ϵ<��=��D���8=͢�=��<h�<�4��Ĩ><ֻD��</O5=X�.���1=�!�<�e��=R<�0�D�S��߽ �&����$���ϭ=�+���]=�$z=��>=�����(<�"�=���w)�=��ȼ �LL<�:<�e<p�+��޺���{s]>7��<}|�=�z�<��<��>AG=J�a=��X=��<pD�T3����>[�=<Χ=��T����<��=�w�S2=`�t;�W\�9=n�E�D��|k ��x� s2=,�>��̽i���6>g��=!0&> !�=����:����ͽ� ���������
v�<��>����j�r�뼘9=�Ȁ���/>s���	Qz=�<Ĺ뼮�{���<��=� ���2=��j= �����<B �=L1��&���^�6�RV�;[��<��� �<?-� ʼ^���[,�
��=M�>X�Ի�i'=���<='ż�罄#=����z<�胼��?�������B��&����s=��5���/��R�=^\7��g�=��=�(�=��8����ei�<ׯ��CZ������7��,q��<�����ڤ<X�=��=�Z�<�y���AмX�E=���= 8<�0�<��ּ���=Vnt��Ӽ.�0��F�=�"i�0L��^_G<�����=��^�SF1��=6�k��K(�kt"=<��!�� #�<��R=8
;�x5=�W,=�f3=U�=O��=�>M^�=�tý��<tY}� V=���;km�%�N<���Xd�=
�,= �!=���=�o����<|��=h_���=6_8=�=�[�=�ps�]Ʊ��"�h�<,��="�=���=��=�珼�$X;��=ٲ4�8�d=�y��*=�"~�U8�>�<��[�ͽ,q�<8&=ݸM�jpy<����u+m=� P<<k:����x䢻C?ͼH)A���2=�l�<_�X=��Q=Q2'<N�<C�f=o��<�+�<��;0����<�o,<@�໡����<|?|��`$=��ʼ#����L��W�;�ٟ:�u���w=��h�Ȧz��y����<���<I[ν�w�<3/���<k��e"��7(�<�nW�Iw#=f㽶Z?:L�j�!�\��	��+�<G�`;�l(=Kk��j�G=�C=	��1�%�)���S�AC ��y�=�����<���1M���t >ʟ�< >�=����>P�M�z����l��8=���=��<;q�,�ӽ�ԋ�Yٺ=�_>u��=��P<��=Ł�<Iu=f��X�����=��>8���k&�L��=��=YKB>�=�qJ>�>q��;kм�����=\(��JD8�l�?=�(���1��¬<2��"c=,�w��&�����=Zּh�=�B��Pg�9_^=MO��#�=E�[<��<��+=�=�sꇽ~<ؼi��+p5;�O�S ������y33=�.=[��<�z߼"���Jٰ=6=)J�)	����C<��T��=<�E'�6��<\ �=3�=�Xʻ��λ���;�U<=�Y�<� ���g�3Eb�|u�O�Q=�6#=V7!���p���(<ј=9��%,�-uh<��&='������-�D1�;�p;�è=���==%
>/l�;ڰ�;�����p=��]=Ge=[I�<FK�q��s���j	��V<�=]�T��A=� ӹm_[=_��<���<<T-�ĝ⼕v<-֐=sN�=(�B<yԼ�%^<�f�<�m�8��<��ƽ��������ߗ=��=P��<Ưg=��<�	*=��;�W���Ւ<A���FU����<瑐�*H��?� <\���*��;.y*=����`=�=oy߼��;=��g<��꼝�Ҽ˹�=�ۼ#�<<g�"�{�G������Y=A�<�����(��=/'��:<�o�ֲռ�_��Hc�;�#�<E�=��q�n�=��ܼ"��;��=�\=m���B�=���=�߫<{h ���p�&�.����Mi׽�x��<��=�3�<���{HU=1�N�f%踭ژ<8�Z=��<���=c�����=�!=C��������;u���uG�<����($�<���'%���=�xv�Ix�;�G�+�ּ�}�ǯ�<���<��4��=?��߫S=Y�����-�'�?�R�`=��=K@w��S�<�{��2���L<4�g�A�;=����>*=��)���8�H>>`��<����mIx>���=�̈́������)�?ؿ���(��$��-�N>�ǁ;'c�=pڼ�f�=����(Kν�g=���<2��<]'*<�==5|��m�K..>>�b��g����= ,�q0�=��=^�>E'�Q�彭#���?�=�=׼���=��K=���=�L��|�%�H������=�M=iW	= ��[�cY�=sͽ����`n�=���}e���u ���%<XG�xؽ"��<���<:O=+��;��<��6=�QJ�<X@��7>��s����~�/�����b �<Y-�=�R��8���zɽi�K=��>"n>Uꗽxf��j�=��J=��?=ヽQѤ=��=@�"�7��n�=k�ν�r>��^=��>��">�C�=p_k�)���4�=C����;=
o)���`<��J��jx�o���R��<���3׆=i߭=)�l=�P=����[|=����xn��ڶ=f2=�<�='�=ѯ�<j�� ���Z�>�'�=z���Z�<�C�=`��f˂=r���$I=�A�<p"�:9�/�_Vg��4ּ�k�%`=�Y�=��y=b;=��;y\6>�7,��9�M��R�����=�I6=�����=�M=�z�<�K�=�vܼ����c��ձ=r̜<芋=�Z=�E=��x<����%=VU	=)�ƽ��n<G�����6?�=�Ii=@�ӻ:ڦ=�iF=��B>���>��Z��1��돽��ɼdK!=[��딧����}>B�7��V�<ྒ<���E)
>̧�ș=�D˼�=Vؕ�Ġļ�ߙ=�e�<�s�<��5���|�<>lj�,�r=��z��=��=�Ä�H��<�K�� f=����	t�=5J콛����A< ��=+Ȋ>��P>\A�G��=�2�<�!��4����N�~�<�E>�Ɗ=𘆽�pQ�o�/�~�M]s>)s�=�lu��{}=BA�<�����aU=�!=���=,�+����<$��<�勹5uP�=�=�3��*ʻP=0JC������[=u�:� �ʽ<!�<8|}��a�<{�,>��<�+���5�	9�)�F���;i���(Ik��/N��%<-s6<�S=��5���Y��y�<�����j4���l=`�ż�_B�c�N<썐=�鏼�FI�D5�;-K|=�-h<s��,��;��b��yȧ���=�v<���+=&q�<,�7=�"�=�b�"����O=��E<O2�n,=va��+�&=�Z2��X��m��;Ǡ�=��e=�c�L�(��'���>�=���2�@<F���]�=)k���K�<����'�����=�P�9��)�բ�N���"�=��>G'�=@)i< � >�R�=|��<n��< �s<�>��>�{��=�<�J	=-����C�=�g=l��=��=��<l���`7��<=	Mp�
p<�����ӼS����+=�Z�J����(�c>J��<�|�=M�B<^�;�0���7x<�T=Q�%��1��*��<umA=��ԼP�v��u���x��Q=��<L�:���<��=�-s�#z=��^=^�>>�|=�gh��X=���=�u�<�"!=�yz�ĳ=4=(�s=¯�9��<�Zm�?�������x��<1�;}6^=oe����<8 J�4M��ڈ=D�r;�eл�hQ=n�)���2< ��<�4t��"Q=��ѽ�C=�|��/>य़<�=�<�t�<��6=WG<c��B�<h��<Z��<���*��<4Ō��V�=����@�7v��=b0���g��=�)^�')9���*����f�<�P=�h�����	��=uy=�NZ���;�R�=�,�=��$=��������J=�=���ae�=4�<.Ml��ן;����<i�<|9�	i�<���Y�=���=��=3�c=��=�E=i�2=J݁=��4�=�~���x,=��x��5�<L��<=vK�E������t�<g[�����<:6�;uY��彼V��=]B�=EX�fK�;����<7,Z=f��=����%=�Hz������ۃ�C/ۼ7�	�k�><JG,=�����ڼ�(F=�U?�_�&��Իb8лS��<]���0����F<%���*�=��=]��=�F����I�k�4�T���{��<��4������U��X$<3\�=/&D=F@�<�Q/=�)���'�=b
�;�����r&<��V�����㉗=��z<��ͽ*~�=�s�1
<Pv��]�=��.,�<�Y<0б<FVͺ��#��=#B�=��1=����<*�8:�M��G�+�c�p�h�����V<C�/�t] ��2=G��<E3��tS�=��<��$��g=<o���<:|�<�\�<aO>I�J<��P<�x=��y=t��;۠< �W=es��oW<���=�c�N�x=P�h<��;ju�=F�㹽��=�؅�@,�=�:�=`��=� �l[<(ֽ�'\��#�kW�=��>[�!=��p�1i=�=;��\O�k���vκ.@�=��~�)#��*D�<E	��+��2���L����=�FS=S=*��)�m�rvo��6U=�B�<���<6>������b��GW�}��<m���u#�6��m��<�3��V»�S<�Ҡ�ȃ=d:��!4<�Z�;�D8=���=�e��W�N�>��l���A=��?=!Vt=�ü�c/��b^�k���6dW=���gh�=Nւ<Y �<G㿼��<Y�������ԇ�<�g�����:���3"=�>X��~�< b=m�սuf>bM�;:Y��1}	=�#�=�f <����9=���;k�=���;"۱���:��7��7S����)�a�o��b<��7= (<=:S�<0�<�J�����!\=s�m;�2h=�`�<�I=�
=�8=�(�H���ha=���S��<�i=��<�0�<�;�j�= ����V�3gμe{ҽ��
����=���=�7�;�=<����}2f<l�ໆ��=��W<,(��w½ﰅ=G�ɣR=��;K��=?��<�i6=>>����Խ=�4�YCd���\�� �<�i<�%��[`�>	�1��!�=��<��i=;�=mѡ=�s=�_̼%�R���_=��^=�<��O��P?=��R��0C=��;��*��̧<���=�G�=�I�=~�z3���pL<>ێ=�����<D఼[̊����=�z>$�>.��=C�D�BW=�������7�e�����Y����B>�' �go��:=�#,<�!��=�4>�/нk/�<�:<`l��`�(�n
轊��;=�D=��=�=a42�m����ؼ��=D<�x=as=����z=�7�8� ;�$�=��Q�_��=_�=�JN=�=�kN=(?O=���v};<�=�Ǡ��7�!�J��Q=M��M��$\�W�����:>�:�\=�9�<D�;n�V�/?O�< �$,�v|�;�2�9=�!��Z2��
�*/=8�@<~W�;0��<��o��&=��=6A�<����#���E��l��=�v�<�Am�u�2��N�T��=>�<YW=�Q#�,��*�7=�1��zbR<b�@:el�;�7h�����FY=t�<,�Y�o�=]����[=���=,ƼANA��ܛ<VA�Uq�2��9�%=[�U�2�����<k�=��>wq;�<�ż�g
<��0=,�=�}
=�sM�/�,�}��=ٻ<�&<[W��ѵ�<�1�<�f�=�Z��rq-=v<&�$';�Yaf�{J=9�	=u�������}܉=��g�Iۋ;�	_��X=�>~J���D<3=2=�B!<c�<�A�=�7<3�,�֝��L���$s���<�������e=	�P�ܽdDf�<�`���b���E<���<�3���SF�Ǝ��٫g=Ῠ<�'.�Q_�=��=��=	HX��=������P�<?��N����y�<�����d�=9ֽ	u�<���pa��*N�=ѹ=9a�=�1b=@>���=F��=�����H�_D�=;�2>}�=	��נȽ��=2.>���<�!=ƴ�;k;�<�)��家<��	�H���B���Y��V�:���=���ww;
�Q���w�B$�=d�<�p�;�F��q�<ɚs��<�n�<��H=��
�̬2���{<x�=iU��W��;�El=�1$=0���:>��l����<�z��㱽U*=�q�=���A��;4��=�m�=��*;	&~=�࿽�gĽa�=�)_�v�<f�D�r�y=Mŀ=��<"�p�~/Q�H�;TM{��=�����='Ep=��%>�l8��~z���ܼ\�<�;�<�<w:r:>�<�m>��ڽm�������=}%I<�?����;,\d��"S<����U�=.�=�H��]�<h�	�[��<U�;D& ;8U�=%��=��<�������T;�:).�<0�����>F�*����;�P�=@�v=�-�7p>��k�e=.�=�y=Oׇ�*�=�7>m�D�}�n<_`�_ʽ%��;r���=pP��1Y�T&a=�t�<�y�\=��<�1��ߺ9=�$�=���:��=��i�8ի<��i=ub�<���=�尿Q�b<�(��p����B�Yˤ�.�+=O������ d�=�l�=Z�_��x�<�޽ߨ��&w=���=_'�<��߼c3�=�*�=���J��=��=���V%��1^��8i�`����+'>/��:���T$<�N�=���=QP���E:=R{=�8{�^��Jx0=KA�=J�=�6 >fh"=O������=Q�97^=�G�=D}��ؐ2��ͳ<��I<��<8lP;�D缭D�!����o=9�B=�I�c�<��>�@GZ=��ʽ~ú=pʺ���<N=d셼qR�<$�	L�<�=M=��=2-�=(h��.=P���>;����� y�[~����</=	hM=���=U=B5��0����=����+'�73&���P����=<�x��*�=�۵=��z={v�<��}���n����<�Q;����O9:��W�!<���=~�=���<R	����<��8�>5�=�5���ܒ=>Z�<;
��ڋx;��_=U_$��멽kYf�}Ȟ�@=�{ֽ�,�<G�d=gR���O＃��=�t�=�޽�]�=�I�=����>�u�	nԼR#>Z]ڼ�Q�=q��`]R=�Q3��V���(�K��o;Ѿ�=����=8�=�揻�D<m�9������	:=����K-<5R�=Y�<�I�<߁:=+�ǽi��=��=�U�<w�H<�<,<�67>+\W<�7%�Pc+=�β;)�G=�0=]bH<Gu��I��H�R����;�pϼ���[�=9o{�"7Z=.X�k߲=㗽�1<�����c�<9�g��c��w�<�ٰ�!Y<g��w~�:�.��c�<;�>κ������$={tD=���<���<Ņ�<��=3�=�^ּ0S��W�<���;�+�M�y=Ŝݼ4a��S�:�N�T<ċ�σ�<?Q��5�@�5�<o;5�켍�<8�W<��P�$|�<l�=[J����[�<��<��>�3�=$�9���O�'5��8=����Q����=O:�7�ȼ��׽zF�?�A�*�;�]�����=Hv,�����k�ѫռ`]�(�=�t��B�=Q�˼D�v�=]�����=jm�<�s�=?=��D������<����~��=�B�9p�� 9�QP�=�����<0��v(=�7�U#$<��`<���<�:.�%��`'=Z=伾&�=�-S;�=�QX<�ܣ����=�-=��ҺƑ">��ҽ^���P�:����w�o�����v_�P��<�?]��yC�g܆�,_=8�>T���&]�=�I<��q=)m;�Vu=�~�� <Ed`=�*��5?A�|�ҽ�Bؽb�= >�;=�&>�3�+�l���=I^�<�������;�i�=��<����9D�>[�n4.<�4�<�/>4��=]e��cSi<w�<�hb>CY˽���=]7��w����<��/���$��z������@>ơT�Ȏ���>H[�b.����#�"�)<������=R���1f�=.��<�v<\ɼG�ٽ��ۻ3����<����dqX��Q��?�4��!%��*>h�F=6��n(>�Շ�9g,�N�[=���<�؅=��=�<�<v_�F��=(�,��Gf=���9����-=^�*=�25�<�=<�ȼ�/�=��罸&L����=�Oټ3&�=�rQ=ܣ�=��=HE=�����S��^�=cuڼd2-:�$��O����K�=Zu�=�%`���8=S��;}���m�=� �=f���o1�<�$I�Js���ɽ�
,���<���=��/���X<�x=�2�#|�:��,��Kg=�3����<h(-<���e�<��=��N<=���Ŕ=���;��G��)ļn�<�1�<�a��[�hx�:N哽�΃=`��<x*�Z�<ig�<F�=���"����C��=؅=�r$�u�(=ɑ{��ѡ���'�u�?=�^ٻ�϶=+I��t�4��������=#��<�k��&=��=Z
��Q�<LUD��i�=ҶM�X@=:m��K=,�w<��<@���
�=ˁ=�����P�<�~�=�{�=S%=�<��~��M�g=����y�=�Z=	o�ZԂ;5~<7�=V�=�u�<�Ũ��� =�)���@���XX�й4=���^=W=�Ґ�ŔK=�۽9�!�	��4�=p7��G��==t	=QZ<�=�wV�=C�L����;��a�I=�=��5~�<�Y�$�<�(����Ԧ+�W�$=��B=F��<p��;��==�=�<u��=F�C��P��h9�}�O=��<�I���΅��y�Yv�����@�=����0��� ��M�=���<py�;��<�A"=}�-��jy���\��ݐ=N��=�&�=��.9ARQ��=5�ֽ��9�#`�y��=[����v�m�<�s����;��=,�Z��ؿ<X伤]=�t=3��=�\<;�.=d�U�!^r��0�;�t�=��=D�B<�IS�X䊼)�=By�=���=aYϼ��s��;�Ƌ��&��<q=Q2�=����Fw=Gd$=� �=�흽�Q<�-��=ȥ����=��:�{}ӽ�$��iy������A=>"�����}�0��ZC=l@�;@e<>�ƽ�<M=ǐ�;L�=�=z�=]
�����:�����J�;�&�=�\7;�8��}�Y=
�h=s;
�z�9���
<�~>=����=P;�=4É=ȸ�=��C=����:e�5=�x�=X!=X =���$��=g�<ܜ�=2�=W�����׽�Uv���;u�8)Ƚ������*)��X:{=I�<�P� �j��;:`��6P>l��<v�X=b�=�e�=߈z�����@5�H����ݗ�p�����;�~%�{U�����=I>>{R���<�6P̼�c��d��=�e�<��=(�R=� /=��2=1�䇌>=���?�<����Db�Z">���;��8��+ٽ^}�>���x�=������ �e�y��(��5�����U�f ����=�B;�����f�:}�x�\�����a��=���z�b���/_�<R�&;Z_6��쫽�P׽�⨻��+=��>�ڽB	B�V��;��<��=D�<Ա�=&Eq�v%�=�伽H8S�yE�:T`�<SԵ����_���\<��i;+ߨ��N�<\���	���x�=��t�������<Lʋ=�	=��<�ѽ~8D<��=����<z>%���,C=y�b=r՟<O>}�V5=G@�<��<�I<=�X�:��<0��=QE�=V��=���<z5����=��:�������j�*<e)�<�0=3�м�>*	��Fg�;�ɼ��׼���<d9�B�L=E�����9;�����&7��V�ne����,�l�
�u�8��T���l�=
��=x�=�j�<X��J��=}�S�5+,��=��<�H#�=#�����P�<�:`�k��R�=E���$�>Ws�=�������;ש��5=wƌ��1O>q�b�	ީ=�=d��;����5�����=�E>;6�<v��<���<��仞=Z�G�v��� ;zJ�!=�1x=(Ӑ�ʻT=�,=�ǎ=W[Ƚ7��<ʇ�<Gs=�S��>�q=����H�;m�%�8�=5䔼�k����=���`����=���<�P=A��=I�=[�>�z%�:�Ԭ����<燔<��\=�
�=5E�ǘ�=	����=<
��?<P?սM2��O=+��9|>�ќ����=HD��B� ��6N��O���`�=�y�<M<��&��M�͛�=�FT:��A����<P�(��z�=�E�����Ul���:�<Iѽ�/��K�>����=��s=�
B=��Q<9�N�({����<@>�u=L�8�ؑ�î�����<^�u�V�z�6>*qX�O=e���"ѽ`j��[1�=az�;�н�������<��7�䱪�� �=�}�=�1M�#�=*�F���>O!ѽd�Y?���=�Ef;%�޽ƅ:=���=�S>���=�T�=ޟ�=3��;Q*E<��=�=�zܸ��F�</��<�Bżfȼ��4�T�^�q����-C���<ַ�=liZ=�
�慊�֒�<�ڃ��±���7��HX<��I�E={� �F�q=�ܭ=���=h��=�2���^�jã=h��<'=��<�Ɂ<j7m�䶥��l>��?���3�L���ٻ0-��H����T�IG����;���<�h��D�Q=� �=�+��3i�=B*���<��ļu6r��0�P��<ۅ��kt=�����=�H��E	>(:�=�w=X�=� ����L�U��|�(*ֽ?=�;�����;h��=ԝ =�Z���fy<�e�
<�=*H<:=�^���f=�_޼x�D�NU���l.y���o�C���8�Ƚ(x����=��8>��=6�!=I�)�b��:�O�<�[`=�����B=�mA��	>�:a��0>(��=s�=[���6�=���=�='���սc�J>�*���j>��i��Jź��RԽeh���<������=�K����=�6]<�f� T3�<��=q;�����}���R��< �'�6Q<eH=����\$�q��r~�<3����H��
>���<a%�=ر���4�l)k��=�b<�`ѽ5��<��:��a=�#Y���"�>����+���b<�5;�di�==�!<�`V=ƃ=�'="�(��½�5>(���ݧ�=,�%��;=w�>�Q���)=� =:g�=vL$=\�>���.R<�=���tF%����<K�[�� �<���<����5!�=�`"����������<����k>�'=`���;� ='��� ����4<�D���;+{,�NG$=��=y�.��� �c�=.��;
��<)��<D^��N#���
��G�9�:)>��>=�vx=�2�<~�@;��:��-=7*=�.�<�Ɗ<��8;�+��k��<Rц=���dP|=7���)z����.;��p�y>���?=�x;=���6Q=�T_����<���=Ԑ��sܼЋ���?���t4<D�=��=��=�ҽ�m=��=Hw�/�ɽh�L�;2(<�	�Ǚ(���,>,k�����a	@=6�4�$�6���<��=c~D�8>RG�:q�ļFy����"<S��<kC｝?
�x���-���\.=Pl�>��>"�<��νyB3>J��6:tp�:��P=/S>7�i�6���Q�=�s���=�A¹7	Y>]�=�3�=!����ż6D>>y���yH����������6�����ν���;���׬0>�8��=�9�=�y�z�ƽM�=`����.��F�<Pҽ���U;����M>Ք����������dw���-�=f[0���
>?�:>��\�J���&'���]�<��$=C����ý*0��-%w�8��=���=�2=�ýi�����<Ȕ�=R�����d<��<<� =JB��+��H!>��̽��<���z�n=���=R5�=F �=A��`�:>�\"��)�=3�ǽ����Y���*(����uZ�=7��2�%>��8�5K�=*�=���<�(�����<�<�=�?��/3�;��s�@@2=Mw=�1�=��������;F�hY�<�H	��p�<���<̆�j$�;C�O�X�<oBy=|<#�n=3��=&b���=P�U=;�=z�]��[�=�=��ټ��0�0/ýErD=��=AgB=];��Ʒ�@��=[1e=
�=�'�={?�< �.���8
�;��e<����r��=�w�<G0><�R��y+�=�H>��=�ޠ<�ҽP^�<�;<Y��=��
=�q����@�^�z8<S�=�-������z����*=�V=��G��.1�(&
=�8��Ӻ;)r�;w��<����#����r!)<����H ����<�Ψ=$Ϻ�Tǻ,]=�����=��
=N�<���KF,���L=�]��^a���u��@�D�S����i<�A�<�0�S4�<ϖN=�'�6�1�n-�<��������+�<���=OZ�̱�;ڧX=@�<�)&� �<�˼��<�
�<�F���RB>>RÚ�V80��<<�����=�=�r
���W�ڀ2='��<�����0;.�,��ż�ݻD�>hX��ԁ=و���<�l;c��EUZ=� =���E\�<-h��˗U=�o�"�<���@�=H!
=�,�$�y<���IO-=�ȶ�f=s�ڑ2=��%=?!�Oݚ<��۽��<��=M,�<�ܩ=�=�=�×<[�l=�`�<����q�U��Į�+\=��}�i�|���W�O�,<���;V�Լw�b�� m�քܼ�+�<�C���w�<B��V<?�y�W�=� ,�?���V���3�\��^�=��Ž�U��02=[X���<Czm=!/=��i�7=Ov=�ě<N7'=�<���M=���=��';9m����=z��;��<t��<��R<�ǂ=)9�<Q ;�
�<�B��"ټn"���N�{3=ͥ��0��a`�=��<7�;<�1�=��|=�m=���ɻġ�=���*u= �=9��<�@<
cG=o�W�v��:����m����;����=�8=u8عFk(<�,�=�1�陽 D��;�n=(��� >	[$�G���i��=1��e���L�<6WY=��j�{�w=�:�<��@<�"+<eL�^Ґ=�/�������
�=#��<F,��-�\<�����=4�<PГ�V`½T�]=ќ>W�⼰Ģ�K<����={�=/��=��T=
��=�Me���=��=6E�_4/=��,�=���q�<��<U�=b���������~=0Q�x �<`���E6�=�4��'!w�ڐӽjE\�r
M<Ϣ=n������[�<�o�=T�̼G9���d�=+����A������D=�)����b=WoO=)g�f]��*m�����p�=}�����=���H�ۼ�+=��2�+��=݁���=!>�;����W�=���=�$����>��ԼO���>)Dt=~^;8R=�b�=����=\�v���<��2=�hL=�ɏ<�P�<�����y�=��<�	>�݄;�żb����֕��w<���:�n�<ӗ�=u[|=��d=�.�\dX=Ab�=��<ܙ/��|z��ښ=�.h��2�=��)�'=+s�=
M������˭ݼ��G=�y�����=���<}6�=���<?㘽_"=B�$��c��'�;2�;�s8����Nl�=@�����'	�<����f��G7��s�_��=���=2�=�`I�hZW=>�����~�J��<߮=�co= ��	s�?�;���{��_dZ=���maU�@0�=�k(=TD�=�+�<��S�a���ΰ�CS�<@�(�W
�d�=�»z���e\������n�����G���>Q�<����v=�$���ֽŧ�<�}1=	2�=."���ƚ��. =�q<��<d�&;|&�<�_��`��=���;�=}���"=���=�U�=ʜ>�h��棽�%�=wH�<"M=��瘼�.�&S�=��Z���O�	~�=?0��Օ^=I$=�BE;(��=�k�=D��<@.�<h[-�ē�� �;��ԕ<gs=���k�̼ס<�n����<���=%3��B��T�=�7<��<-��<���=��˺���<��R��{�cto<�.�=�lϽ�׼�Vü�U��	޺,b�<���@d�1j�=��=�1���E�=�C<�t3<E��E]ݽS4�:��=��ٹ1*=m�h�;�<�'�<��U=q�O=�>��F�<k=���<x�4�Fϖ��?ƽ�Q =��=���<B=��zǻ�R[=M>��=-�-�c=~`h=�?=:��(�=���L=4�<�ް=a�"<�Fｐ=潫��Id�=	��,s*<w(= f�;�6<��>=�O�"���=;'��f�9�";=_6�=�<�I�<�iO=�`=Gn��;ܠ���<:�.=NZg�,��<���G�ĥ�Ii�=��=�|�=y�������w�=HF�<�!�<}�=Ŗw=�4���K�~��c> 0=4k =����S����߽ � =d�ɽ�T�=~s�=F3ܽY��=ռ�=Ez ��8�ʠ��uU<���<�8H�sFE>EJ�ə�<!��< G=��O��R��:�'>G����5>��(�A�Ǽo�/��A޽ڏ�;_�#���k)���ν~u=13>�V>�KN=��A��ڦ>۩�n`��X�H����=�%�=ժ��s����0�=$}��>�t�=�C>�=gq<��=��)<YI�ڢS���O��
����<�{�6���%�l���E�;0��>Y��;U�=�p�<�7<S𗽔'M��o>�J���$��nh�&c���S,=�S�=;�o� ��=�4F;2���ճ=L��n��z�<��B�<d5�<���=��~�4�=m��<�H�<�!�<�V<�v�<��=ȟ�:x"��Tq0=�҇���=�<ễ�Ի�w�=�"G=U��E�=�T+�z[_���D����H�O�rf>��<�Z�=�>z��=+2��S8�=-;�����<_n~����<������;�L�&=�(�=�<���.w&=��`��꼔��<�Ub�xM=\&d�H-1���=�\M:�T��n�=��C�B�<��=�ة<t-=\D=���<i�8��)�F[鼷,����r��#�=־�=xB/<��k�ȆI�3e]���{�f��;a�=y-���x��!���;L��~
�������lf�
6�={�<�����=����5��FW�<�I�<V��=mi:=�t�;�==��޼���=���2F=zDP��0�9'�k<߈���n�=l<v<�H��d��o�����@������6֓��t*=������=��/��p�=�ӵ=�F���c=&O�����U ��=�w��u.=����ᩖ��vF���=��+=�e��(�z=�&=Ԩ���G$�+��<\��;k�X=xJ�n����/�s����s='�#�o�$>�
��� =7J�+�<�U)=�`��h�=
83� ��>����=>L�<C���?� [��ђ�����x3���%��>�=�I��{�%=�>����4�S���<��W�i�L�φK�`G'=X'��N>�ȿ=RW:�	2d�::$��WW��P�=�F>���=����<b����Ӈ�Y�9������=-B�=�=����/<������I<�F9w#Խ[���s,P�k���.ƽ�K�l�=`E;=Ӵ��<�U���[
>��pD�}Yp=@u�}[</�����=SE�=s,>�F����!����ZC#��&��2�=�;9�����ç��N�=�h�D����~�=�V)�-z�=�����1�=���jR����=6�=W@���x=M�5:���<3��y���J[=O�3=��u�y��=#�<Wv۽ЩH�e5������1<]�E=V�s�վc<�H����-=	����<*P�^��L1缨���ػz<��;��μ��&=�QӼG���֖�=�=�<�4�=Tx4=���=AB�;t��=˨�<�R"=���=C��<��<b�6�.���h�=e���(̼%/�=��X�7m>�n=:/�!j ��Я�Y��<:��G���s@>m��a�h��Gf<�~溤)�~b�����=��ڽ��>b�=#W�=P����+�����"3�{�#=�H�H�=�>�>ċ�{�伥� >�W����^�Ԝ=�)/=x�=z���`%�M)����x<��=���=�t�=��=FR<=^�<���P��=��W���%�%0;��P|�4�U<	M�kS ��#l<��'>�d=E�@�dq���&#��Al��5<.�=�=�=��<�5=�ӑ�lB�<j˝<�Ԉ����t�pu���3=�aI�ޚ�a+Խ�����v��fn =�M8=�#�^�>=�����6<�i�<e����=���=y
�/�L=D�?<�&�E���9̽��e��~�<�,�;_��<De�=(�=��M,�=X�Ľ3��4�q=��n=3\�=���=p�
>�<v������]�\���R
�<-d@=����`����6	?<�i�=�h��=p��<4,�����<�C�$�+�d1�=X�e;��ߺ"z��䛽�ǥ��{�<�ǉ�Z�<���=�����=�����(<�[J��4;j��^n�<B�	=�r�`��R����;�0I=Ɋ����]=��=V�?���<���P�_<���=*4�<|��<��<���<Eƃ=#���jK�i� ��<?�q=���>�<_{�=ެ'=��v=~�Y��U�<�u�=!e�<���ɣ�=Lg���>�.��=Rf=fj��Q���2�̪�=���<�IH��o��mڳ=(���i�=�=�̂��M��`=�z�;����j	9�ư<M7=��=)����=I����<V���K@�܋m��fH��	U=p.�;գ<fCq<D�=�[�:^�;z{>GG{<����=�Ģ�����n���I�����<��<Fڋ:�A�)o�����HuD�j�=��:�2��<f�=\�;�_��y1=�H���=TI<����ͼ��0��,(0=���;
j��k��~˺k�̼oA'��}=�]�<ݓ��5��?�,>��h��׽���x��#h�=>���k�=��&�=��Ǽ&=f���a^<���=a(½�� �׶P�����|Ҽ�E=�6��0[5:'hL=���f@>>k�m���^�Ӗ���I;��i=]���(M>A�/�e#=蹼�%�=�C�=ʹ��V�v=I5O�d�t><)q�~}X=�f�Gj�<K��E:���۽t��R�I>��4�=��G�H�\�^�����R�K�Y��ܽr��=���<gݮ���d���F=��<��<��E��oV�A�=�=%�0=�R������z=	�!\�<�YļE:h���.=�~�<Ad���9<4|=N��4=�I�ԑm<�B�<'M���U���;��@����'o�C��;O�<H�&=D�F�����	��=��<�Ό��c=̽,�Pw�<�}��e�#<�(2���=���o�a���h�)K�<�溼HJ=GdD�N�=S�9=ŝ���N�&��=����[$�]ۃ=tC=k@��׹���Ի�,>=�A;�[=rE<߅w=��q<b��W�=�3���/
����<��<(8=��3=$l:
CV=6����k"�z�
<:)�=t\<P�*���Z�(� =���=R�d��>=�7D<��j���<�F��S*���R��9"�R�<��K=�m��>0�T��\ =V�V9uX��3�ý��;!σ='(o<M_��!�<�6=�%�T�<'�=�ŉ=-��<��>6�<g�ʼ  ��v���� :=�Ļ�Fl<3·=�}'�Y�;����8q''��]��%=����b�<�=����^�[>�挻S?�����<�Df���C=���=�%Լ�H����;�A�<�G��h�+=M��=ɘ#=ɉs��>=�W��8̳;b�!�*I��1�=�@=�ɽѭh�o��������<�(ټ;����jJ�Z�<=˼�!�������.=�<�CZ<]-'9�	��K*�jR�=��=�s�;ߤ=X�\g���Eν�w='��>{k�x` ;�M���/>�m�� u���}b;2�2:��:-=��<�e���v��I��=�_H<�Ϟ;�ļ"=��F���$�V�}(�ȏ�=�eҽ���}��=/�>J0м2c5=���;��w��q=�+�� ��= �=�����bI<@&=���<y�=5�<���=9u�=��=��=1���vv�����} Z<մ���9#>�
>�<\�9"��F��7I��C����<�Qo�P����f�{���d��c��<���>m�e:1������;E!��M�>g`N���S�W�J���[>w褽p$B�=7���:��ܫ<V���'��˽۞��c�R<*��=�y<�-����R=��8��>Φ��l�(�,xA�p��. =*�w�/π>W;H�*�2�Gَ����=J�=gQ=X��=���P>����S>#��D�[=}����;�T���n<�@]�29>A�h=��n���=%�:K��ů����.A=h|j=A!!��B=�y�;�r�=fA���hý]p��
IB�ji~�f���֭��,�1���@�ط�=K`��gt�<�/H=����4�=��(�"�;���ap��wy=Q�F���/�Ƚ�=
>�Q�=2H�=a�Ƚ{����=bs=���=�?� ff��:�X����>v��<��:x"@=�\$>�ٍ��˦=�ɛ�y�<rk�=��Ѽ=%r=R�F=��_�jv�<�r��:&<,]�=l���[�<��>��F��˼Ex���սG$º杊�+yP>+)�tV����:����"G�(s��g� =�c��+>�_+=6�ýjv��񲹽A@
<'�޽�޽�G������	M>�a>n�X>M���񤽏ŋ=�3������/��:~��<�q�=���|�����>>/�����=���=��]>���=mWA=�K`�G�+���/>;���~܁<1~j�Ὤ�#;D� �\N=�e׽�C�>#P�=W�����߼��z��~K�:�<�j_�W��w0���h�=P�	=������T=AVۻi�<��x��*�=��켩������K�ߏмx������<R~=���=&)<%�����=3\=��Q= �4=z�U�+3x���<�?*=���%G��9����=x�,=�.ýZn�<勺��=�O�<O?��G��!i�=E=R=���=�ʽ=�.>�y�<��<E:ݽ�<Z�<h����7������Z��P߄=O��<Cι=���<\⡽-1��K�0��𗼽#��!�{�X�I�;�Z����<�WV=:����*�TN����=�5��X��o\{��~r=�-�;��<A��=�'3;�=Լ�O=��	�o1b��z���Q=P�;��R�,f���S�.�=h�	��6�ґV=���=��`=�5�<Mփ>����<�������=1>͠��ú=�5Y���U>eU��>���sN���^;����v;K��q��,��1|=�ː<%X��92/>pa�� ��귇��Y�=�Y<8E�=T�Իyv<��k=�
>�S����U���z="��<�����*��"����b��=h��0�=*By��v��r�(���+������1C=TWX<  ��=h��=�(=k�#=9q��Xe�=M���vl<-�=���=�����=Ⱦ(=�n�;>6�=�y���ƽ�NP=QԔ=�>�#�=��=?,��$�=><�����0>��	j�<縞<LLü��=��<��L�t��=*M�<J���ٗ=�~�=�/�.�=�E���j��e�S��<�||< F��W�=�|=[�<����xT���<M@�<�K=����p=j����TS);=�k=�2��bXt�8�;���<,�>��fA=�w�=:�d;����=u��;�n
�`I���@7-<f��R�/=n�
K�=F=O��=���Q�h���=��+��!}<�p�=���N*��"}(���<=�z�=�'=`����r�`p><E�9}ސ=�o��>��//�<��<��=���ڥJ��T=!�$AE=�"G=�7�<�B���%��''��x�<,4p���L���<r�>����P9��j�=o&
�7��<�l�;~��_��������=L� �[�=�'/=� 3�-��=�+�c�<*��=2���ꮼ�	λ�ڮ�CG�i��S�&>I0�uܜ=O�=�ߖ=�؍<�$׼��=O%f=��H<D��5�<D�d=_���]6=������< g="X�<�͉��@����<l�	<���<(��#��G-�=��<Ō��Nc�<L!!�Ї��{�!�M=� x���.=��MF�)g���&�=*���5鼾��=�ߏ�#��<� ��c>��}=�h���^�<��<�Te��7�Hծ=�.�����C_�<�+��ڶ���2=-F=������=��=��ʼ�}|�?m���żC�����=N=�B+=|��=k_=
�5��{;'=.�H���;d�=ʺ��؊=�m�<�0t�D꽺c�s=���<)�潶~��]�q<#��=��=�B>;���<7��|4�B����r��<��h=;�>@ͼ�RR��|<0�=Za!<[���3?9��v=i���j�Ͻ���4:"�Α�;���=��;���=�	'���[�XG >�&�=��=9�;����<�81�ܣ���������d�5=W��*:E=N;u<��ֻgs#=-�ͻ���;Y���e��<��Z;i�$�RD<pdw<�����+S���Ȼ�[����`/=(��=���=�(B�4���	]<�1="�n=�2�{H���%=��{;�FH�%>����I�#��< ݒ���S�c�����5���U=���;�3=o���j�\� �=,��<�Ň��:s��q�����ͺ!<[�>|��}�i=��˽��P�¼zѽt�==:�<M����$ =�T�=Z팼ˣ�=F�����6S�=o�<M>H��=c4�<buû �<�k��o����m�+2"�3�A>�A���B�!�T=正��ި=͡�=�J�%��=>3F>��Y��= =�A�ֽJyK;;.��ȡ=9�)�O����'���<)m
�HXŽF�=�����=Lť<'���\ż��(�W<%+�*���;��ح뽩!->��7>�E>�D�U���w>$3�p�F��<Qx�=,��=�C=�2�n2?=�Z��">`L=�qP>Ek�<N�=�z����K ��MC�X��ӗ^=�6��k����BI���3<"s=if>n�>e��q�2=R�p;}�<�c�9>T=�D�=�b�=,ɡ����=՜���(=s[�<��N=��=��q<-.l�������<�w����e��`=H4��J=��&���&<�$�һ�=$�ͼ����p��5
=��.�V#�=ն��3�w=#�E:�B=�g�=%��q�<I��<!V3=H�1����<�ru����6$���4?=7�q=�!A=�<(�g���>q@�:��R<�(�<�0e<HA�=D�ouۼ ] ���¼���<ź�o��e�z���1�n<��̻�-�<F�;q}��q�<�=��=�����Ƽ�|���ם=q�ͽL`��9'ͽF�l;+J'<���<_��=���̺�=����ý�Ln<yH���u˼ȴL=H��<߯���U�=�"�<��N��ˀ���=���=�n#��s4���<�r�=	�H�n�z�mR��6K=�C�=�]<���=F��=���=x���bKK=�,�9�d�pL�=D��=����=1+���<�^�28
6StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_6/conv1dConv2DSStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������@�*
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
:����������@2S
QStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_6/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_6/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpConst*
_output_shapes
:@*
dtype0*�
value�B�@*��r�;�	��޻��NUI�X<מ5�zˆ�|�v��ǀ<����<�l���V�J���K}���L�4	м�:WY���ݭ��N-<(��<�4㼅`��--��A�m�<d�:W)2�Xb��<�����	7r��<��<������_ɻ�ʐ���2%�bʵ<JH���*�Xa��P*���<��H���;*�Q=B��<:��<�/V=��j�; ��;�=�B�SK�< 긻���; d�E���2;
9StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_6/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_6/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������@2,
*StatefulPartitionedCall/mnist/fc_6/BiasAdd�
'StatefulPartitionedCall/mnist/fc_6/ReluRelu3StatefulPartitionedCall/mnist/fc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2)
'StatefulPartitionedCall/mnist/fc_6/Relu�
+StatefulPartitionedCall/mnist/fc_7/IdentityIdentity5StatefulPartitionedCall/mnist/fc_6/Relu:activations:0*
T0*,
_output_shapes
:����������@2-
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
:����������@2/
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
:���������@�2R
PStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_8/MaxPoolMaxPoolTStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*0
_output_shapes
:���������@�*
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
:����������@2T
RStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_8/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������@*
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
:����������@26
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
:���������@�2Q
OStatefulPartitionedCall/mnist/fc_9/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer��
6StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1Const*&
_output_shapes
:@ *
dtype0*��
value��B��@ *��)��2*�;��;��	�W�-=�E����������=�g��|k���=�� =���tiƼ\^
��f���=�ѱ<��6��<�o6=��=�� *F���f��	\=�t�<�p����&=c���A7�=�=�}ʼm�2��Wv�l��=&�"���˽L�>uߢ;���=�L���a<�#���G=j�=�2=F|k���};b⍽��= ��=�7=���=����_ܺ�4b��Nx=�0��`���=���#��=M��F�:�^=j��x����e*�lּi��<�]= �=0�E���>ډ8=�-ܼ�<��=;�;�=�l���<^�Q�U�M��=�GF=x��j0ݻ�8�;*`�
T�=�7�=%��<�Pz��̄�_=G=}�N��������ӛ=�|=������<��[<�];3f��>�j����<M?<��¼��<��>1���7�<�V�����`}=�c�=G	5�1��j?�:�S���=��>W׌�~�̽�b��!0N��/�n=��"��_�<"U��L�
�EX廠������m=��^<[��;R7�<��Z=y�=ީ�=�-�~���5o��\<����3»�m=kc�����V�ʼ
�����>eϬ<!��F����;u�罚>������<
�-�����b=t&3=o1����$���a�ϪG><y+Q�s�<If�<����� ����	�=�7=����*�g�~�(��g��N�<��+<B=����Fx4���"�i�ܽ{1=�2�2"S�\͖=���?��=N>��)=���W���t8=��[�T�=�5��^w��}�<\0g��(�=C�=V��6�Z���9=�ȹk����׼(l�<,���*�=!i=���}3=ԱZ������Sٽ��|��y=e�;��ƍ�H'�<�Q��8F�>�;�׉�-J8=�M=�(<'�<�nD=W�=����D֩���,>A)\=�N=�&�=�oi��ʼ�k=j�f�M=N}y�Fs�<`����f���9=��%=��5=g2�ɭ���dԽ�x��㨽A\��hH�<���x	;��:<"���ը�/�_=� ;��Ƚ]Ž{�@=�)�==u3=K��<  �<�Y�����<�u"=����/N=>�!=/N<;xJ'���켂p��q��r6?�:H�gP����<J�����<<Mb�<A��=*q��@T\<g v=Ŀ�<�|��╼���=�+�;ȯ=�w����<4#�=�����j0���:(�h�gz�l�;�P <�B�=��G�YE0=q�g<S�*�]�<W�K<w���I=f��uZ>~��D�= ᄽ|]�����������=�Mc��T=�D��Q=�D{���/<D�i��>�=��S�穩<���Z`<K{-��@�<{Fq;џ��;�:!F=W[��8Wi���>q�e��Ǵ=P`�=o��<�����5��y�|<[o=�9�<�AǼ#�<��Q<2�B�������S�u�=�`�fu���<��`���ܻETt�Ƥ,�$��=2�=�S���>GS�<p�<��=Pa�0��HD���4=�����Q����<�г<U��}�=I?=ZY���p�֋�=NR=��2�VÁ��];<De��{�=MT�=$�ûl��=4Nc�U��ڏ��G+�=���q'~=�b��v�EN��2oA���=��3��_ѽ/yɽg����ü��<�_^����`=�:>F����~������.�Pp��z
l<�-��h�Nu������M�r�;">5w<�E�����'<XN��mN3=�N<G@��wS��'����U��;H�,�u��u�������7=i�7<�J�=�b=]�����=ZM=]�Z<�Q�<%_K=λ��&���츻v��<Xe�]\�|�='�<��;)«=M���am�=A/=�����<�Wٽ'��"eh���:�TA�=z���9�=d�E=�	=ё�D�*�R�;���=��=\�=D<=r:�Yp=l/%=$ˉ<�;E�9<Mo=��ս�i��4��A�����<�x(=?�>��ʼ��ϼ�����H-��Z��\���m\ =��=�='��u�(�p���%�_=H�~=3�����_�r�׽�z��
�=�V�=jK̽��>�.����ɞ��1����^���������<
�=��>4r�=ެ�=nm?�ù�=��%�(�=�Lb<� �;�n�=�T(=�z=X�<D�<"\�~��=o;*����=Z���ֹ���=إ"=`�=�ъ<���;D�<E[=��<�9��_������f�����2I�=�|��ʌ�g��;A��3C	<w�����<�xI=�/d��k=}>��B=����~<z��zZ��~"�9	�/�;=h덽\{���<�ꚽ�{	=�%E=������X<}��bc��� >�1i=���<'Nu���]; ���e5)��yܽ��d<�R<?$�=�>�g�<Fm^=Nb�=�:
���\<��½ㅽ�+��q=�y=�ï��Xɽp�����'��}M=J�����;*�5����.0�<-G=�d�a=�o�������H\���=����o/���_=bYT=��<+�ܐ=�>w��=�ɓ������픽�r��^Ӌ=�7�;����v�C��K�k'$��-�9�`����V���}S=$���G;t}6<�yq<��<���=�,<tc&�4�R�U�w=����+��M=�^E��|����(;��=�]\�P�<H/���򃼷���í�<��j�4����=ݯ���$�=3�=E���R~*����%a=�ս�K�=J�==���蕽[��=��U�Miy������p�:iR=ȏ�<�Ŕ<W�T=��K�6��Y���d=���<k[�<"ϫ�[W��O何�q="��<�]?�Y~=5J-<}�=�l���S�0#��}�����u=�]��w�:.�0=6��=�j����<*�����K=Z/z�k>���x�k�P;�lW��A��%�κ7Z�<�6��=����9�*�~����<D��p�=�=^=DV�����<̉*=�^��8�=��=�x
<�P�-���6���I�;7����0�����-��V�<�`/�_�6��&�<�}ü-a�<|�����z+�:�;5=Oޔ=� >$v�=񷇼�y��� %=�ʇ=:�W=����G�Q꫽%���qo&={є�v��N�=��2=y��<ɑ�=�,��̝f=���a/-:�H�<ߊ!���<�/𼒩�_����k�����<~�b=_%�=�h����.���h����=i��;�"��ν�/-���*˼9l�<QL�˥$�݆3<��Q<���J*<>�=M�;̾b;�X7��l=�t�<@����8�VꝽ��:��;�i[=������m�.;'��<��B��no<s���~k;I�Ƚ`�H������=gs>� ���7���m�c�ݽ��Y�4���e<W2<�8==`B��щI�ă��}�=�Դ���@<�����P[;5�l=���<'�=C��<3G��S�b��=�y[��R�<r����;N�#��Ac=��w�U�弃+�<�!�= ��=M�����X���ف����=[Z=U��=��	=Z( >d)<��e�����< �=�2���^L���x�<��<�'�=mxO������V><r���O0:�ZVf=}S���i=�Aj=�yt=�$��fF�=1Nj�����+�ý�z��k��w�7<o|ּX�
=�k<���<�r��>�}��80�Z�"�Խ��<_��<!��\!���/<1����=�9��� ��Ȍ���漗���'�輧�6��֟=��l�M�������=�G���%�<������<�w�<�<�=.�2�g=0rʼ��v���~=�֋�	/����<��=�4��}N���=j�<�U�<?�=p�~�Cؼ�鞽�8<)ĥ<�3�������\�O���&�=�]\�ٮ��n�!"m=�H��4.|�:h'�n�$>��=��*>eX�ۀ=�T<�Qш��@����=ۖ=`�����<�����#��r1�����ƴC=�,�<J���������e�=U�����2�3��<�0+;�@�=� =�W��DqB=v+ ��{��"�=���<k����gz���<�O<�n��kG<x(>="0�<�
���#)�NB=yô=�b��i�!w�<�%�6�f��K><'��;/�6��p :&[���C����Z�0����2}�����v
>"Z=��R½�
���}�=��2�k�0��ዽ�$��1���Lp=H����	�l1!��/�=8��m�2=� �:ӴԼ|�%�κ��߆��}�l�JEl��u�:LP��m�ƽ\�g=l$6�<�໾�=��m��=�SE�vc+:?v�<yAu=�Vb��<>=�❽"�\=>��h����3�;�c��=�4<�մ�)��=�����< �T�1<cTv��㞽�y�Ȭ,��m��͙��4������=�P�=�š��Li�^;��
�;����{>B-��'����p�����=�0�$����9�&����6ƍ��W.���7�}��=&W6��a��r�A�=��;�v���3O���<��⽟I�iIA��=U��6B��L�v�G<P4��{�<��<畼�>����R=7�<�iU=604� ��qh=QMK��ㇽ��Z�-ᨼ�Ң��r�<�;"�S>lP˼��%<��.��<�N�dZ�;�$=�ӑ=<MZ=N�)��e��DU�N�:�3��=�KW��v��iѽ_�=h�<ڕ�<59=���=k�<�=-=/�N���(�k� -��Ú#��h�<�?��4�=o�q< y�<B�7=��=ĖA��� =��=��=.Q�<g�˼6#�<m伏�V�S\��/9N<_���ׅ��t���=(��=�n����Ľ��>=ۡ=���;z����<4����P�;Ӹ�<�˽���=��½���1!�@�:=���&���K=8*�=�C�=ea��꫼�ʼ=��c= �;�''�<r��;�q�=Wk����=�j�=?yŽ
�żI�$=<(���T�̞h=�F��{���:��;�NF=��*�1}�06���w���=�=U�ɽ>,ݑ=۱�<�KE��r0��E��:(
��.޽V�=hX9={#<�Ȧ=7"ؼB�!��d�:�U;歿<�Ϳ�0(i�I��;~�K�!��hō=�[���$'=��=/p3>���:��p�)����=VAO���8<!{��N����<ú���;W���<�C>0�|���P��鱽��q� ���
=��=�Q<ڙ�;l�
�1R�����=��2���#=pNh���!�O���9ѽ��J@�=q\���^�= ʻ<���=E�����<	@K����=@�;>�<����W����=Vi<=��W�u��<em���.>!p�;�����½���=��-�;{I=��i=�Ҙ=���=����==�ŽVm��A>�w��U���֗=մ>(o���V�)����<)U��������=>�$Y=�d�<���=��_=}Hƽv^��l�:�x�ܓ
�]���JBe�xһ�oN�;s����p�da�<��O�=�'���_�a�<wsj��X��od�<��D�H�����u�[��;f1ü����==�:H�;D�lص�ĈN�A����`��W�=[��<����!=T?��7?=y}o=Ϋʼ�{
���ԡ�<!����[o�#��=7B��x4��4>Tl8<*8转
�<��;T��Z�;�ڻt>�r�<ǃ̽��P�1�B^=�2��̸�<�t�<�S���¼4~�Ἇ<���=x<~���\�={��8�w�8 :�k)���q�<xt=���=�ϐ<�� ��E ��l���í���=5�8�,g�9S��=�>*d�m4�)�(�����S�=mݚ=Ҩ<���{A�o�k�*O��X�U>Qb
��$�������1=�AK���=9$1��e�=�EW<n��r>%,O�tZ5�.k<2����D��M= ��;`m0>��6>��P���ʽ�7���(�<�\��7	=�Z���q�`�=�Vy/={3�<��?<3"�<�I��"��L�o�����
m=)�,�d0�=,@�<rꀾ��i��$�=�G��>P{������Z=����j���"=m���=�9���h�<NǼ�d�=x��SP@>|����<�
;�eչ����Rs�ŋz;�_�<�7��Pi��x����߼�T�=	K���ƅ��|��X�k;��&�e�?��k�;W�c=JÝ<��]h����=��>4ǽ�MH���}<QHg=�J�<��k�߾�5E%<+m�=+�<��=t|��
;h��+�P�ka����@��u���d+��?�9��Q���=Mu�=�_����#��TG=�Bw<4oJ����=4��<Eq�=�Ƽ�p�1�ef`=��<�e�<4G<�i�=p�<S�W=�둽� ~����;���{&�=�Ɉ=�Ƃ��E���S3���;ĵ����]=��<*��=��=� ���V�u�<������4��+sԼ��=n����l��yw�|��=���;��d��$=��� �=1ϼ��b=$�Ͻ�']=̧��l圼*�=�ҽ�m�����=��̼{1S�ɥ=H���������<J��=�E=�)� ��v��=�a>մI�fX]���;���Cv)������=���,�-��e����=�J�1V��ا�����ј[=�x��a=������)=��<А= ���٭��0\=���c�����<�9=�R�=m�3�ˣ����=��<:��WuT�[�=g$m=r�=�J���Ľ3�;�˽-If<��������)�ڤ���u>cC�<�3ɽ5\�=�ٻ��e���=9��=���!:m�������C$�A�?=��c=K0>=Ʋm�������:�uO;���=х<1��=OQ�����@�Xd=�[	��=��#=e�_���.�Ž�%	��J?��ť�y��=v`�j>�=x=icŽH^Ǽlc�5 =�S��Z���1>�U='̋�c`��D�9���<��2������Y�����=Z�6�M���w�>�si���?�M����<�0�<PKZ�fd=P�_=@��<�^� ?�=؜ռ�m�<_�3<��ü�5ؼ �T��J�=��#����= z]=6�!=����<�'<e�𼀧�</��r'3=9������b����Ο�=z��[�)='���Zb=�5#>��y�܌�բ�=ٽ�*=��|=
�<="��~�=y);>6��<O���߬;�i�;{$=������֢�=�c�=�ux<�Њ<!��<�\=����]��,q����9���������������8���<Lݽ�>V;g*�=e��<��+=D��� �/=�A0=�"�<r�������\=����ﺽ B0�7J=(.J<\p�=ؤn�oH}��2�=YT<�@=��������B<l#���֒�G/'=�=��2�X��Z����<��)��GF�v�;�	>��L�jh6<w9=@~ ���@;���=�1n����=Zr�<0��=���Nja������<�B��|�-=��<D���o�D��p'=&�W=��=�:��P�=*�f=A�;�;��:�d����=�!=�ۍ��u.�D��hF�<�RĽ]��<�i���/��0j<d�@��� <��g<Z���;��g< \h<���o�0��Q��"�<k�i�:�q��������X=T&>{�S�<�9�"�.9=�;���V?<GI�;6&�D<%Sq=��>�ߧ<�32�m^�ߠ�����y��A��΋D���;\jV�`���_� >< ����<\R�����'[J<�)��������=����kP�����=V��=J<B�DL=��1<T�ս�y��� ��C�=��<�ۡ=�����<&�ƽ����#�N^8;p����q�<Y�-=R��;�N� r�1M�;��<[�=tU*���4:Iw;���x="]ٻ��� _>�T=��U=z[2�gs����};|[`:2;^������<.=TTp=wT@=3�=ڿ�vg:=R�r<)��<�Q���Pa<I����d=�l�p�!�]�!>��R�v�=贅�F&�<Ŷ{�x;;6s�=&D,:��=27��>�C<��ڼ� �X�4�wx��½����.��=�U)��|���<�;��l����Zb�.�V=Ҕ�=���<(�=��u;�@�<���A,��L���x�<P��=��e=zS��˼�B=�|�=�t=��8��<�� =B�z=�\üEG	�;G=�`=w4�.YQ<�i������`=�=�=�T5=~��<[�[<�P�p欻��S��C�ByʽE�*�Qm=,:R�X~�2_�=����l��Y�=S�=!��=���<�Qg�Ҋy��$<Ԉa����;U��=��̼&v$=4=#�"<�r��,�Y���S� ���-߼���=�yO=��M=�� =L꼽��5<����x;8��'<��`��㫽��<�<a�_�o=��<�1>=w�u��/� b=��^<�*���d˼�C��]>%�;��;�`u�=x�żE��=��d���L=ͱ�<�C<=F�i=��K�6�=U9���,N=�����<=�d�=���W��Q����v�C�<.i��-=�/�����OR�R�������>eS��:�>;������<A+�v	�<�%��<�=U�ɼ��½����uC@=p�O��v;��<�c��:O��(����m��=?�>�u|=��<�� ���/���=A�'���=�<�<��X=�h�=d���D����~<6���м��=�����_=j��=DP�;5�7�R�A�6<M����i�bk�P�n��h=����@�)����$݂=��ŽL�!���^�J8�=(Y=�10�$�;=��ʽP��=E W=�"��/=��q<kY�<�0�=���=5����I
���ּPѼ����r�����)
�=�<mt�:�c>�#��/��X���#�<r.�=�Z9��.�=,�>�=Ɋ�w�d���<��<���=�i<�=�<Z���_�1>P52���Ÿ=)+���/�=���=*��k�<;�&�=�}�;�f<��k<�R鼖^"��CW����=�\��|�<�MS=L]�=������m��A��<�r=���<�G����9=Uc���6<���=O_3�	EF�D��<T��;�<�<]c���2����;�v���(>=�	=��n�Ic�<���ㇺ������M���,��=�D"��}��š<�"ް=~�=D񖽡n=Gj¼�<R5}=�I�<��˻�W<�Vj=>VX<D⩽��r�HI��I��ӻ�=�)�Gׇ��L= K��{�+;Nx��69=t[���y�<x!���P����e�=&6����+��[��_�%=t��=_Vü�E;��\%y�-H�:�6�ԋ#�u�;g�Hj�H��=�F<��=��=z�;�[=�-4=R@�<��R=��q=%vs��6<ܠG={��=��=��=�8�<�}�<�ˆ�u:�����<@ĉ=� -��j���eU�x=�=��<!���X��=`�m<t���j�_�=Tv�=���<�DM=5j�<#]�����dAH=�n*�x�X�����V�L�����ν$g��2�����ϩ��(0�x,�=�?6<L�=�k:���%�a��m�<��=�`=����&�}�������|���)e�x�B<����ZD=��%=��F��:�������ͽ�K�ߔڻ�Tv�/H���]���Y��㼽�y�=P$!>�N>���=��<�軧�2����X>G�=*4�= տ�"��;���u)	<����8<<u�=� >�
�=E<j�~�<���*=��\8Q�ü��A�4ff<Ԣ<��>�z�;��5�x��f0<p�z��$�'���W�����<�,�=�� �vA*<�e�C/��c=���<Yc��,f��v���v�p=>yg;�{<��:�׽�����@i��ݠ�x8c=m>�=��O�=�|����@�=tF^<(t������Ƒ<�$���?=�'2=�Ql����=�P���{��5&��͜=<���9�����o<P�ڽZ]S�>���q��c.��Aփ�i�.=��b=;>t�_<$y��	<E��=V��<ڶ�=bU`<��ƽ���	�����2��J���~�=?.�=qjR� y>����������+��<e��%Y>�E�<3��~r���Z=��"�ʴ���ʼ0�F��wO<���λ �~�ؼo2ȼW����=�c��0r/�a�:�gI;�M�
�=�6?;56W=�5=rp����7=�	�>3���<6��=G�X=s�=�̗=��J�<\&�<o;<=@Y���=(8<��=�q<J'�<=�J�q�2=�����4���8%=nx=��=�ǽ�ow:�q���;�?��T ��R����]�>=�w[=�Y>(U�<�3<�D�=Z� =˸�)r�=����*/Ѽ@�O�y�<�U��vI�� s��h����$;�!���`d�
`#�[�+=U�
<�:�j����j|��M=m�|=�Ѽ��= ��=)��˳=C��=�� =g}+�M~�;'��XNI����<�����(<�΃=v'#��Wļf�/���>�>8��[����5��O���i�=?#d=a����#���ĻQ�Q��K=I��4�==��b=Ҫ޼�ѻO��e�����3��,��;� =:�<?/;J�[��\�;����_϶��7���;ԅb�Va'��2���r�缁����<q@o=H�2=c�=�"=Kѣ�D����h�=���)o_<����=��;eQ<���=}�����I=��A=�s�FH2��2�nGf�H�1<QӢ=B<=�����Ͻ8�<f�=��=,�ͼ��b��<%=	��=~`�=zּ�5�9��=��(������n�_И�q�S=)��J���������=�9>;���=��=���<���;dI=qwQ=$�=j�<����et��ڽ����<:Tn�=<��<=�z��㻽Ç��o�<���Q�����=AFh��*���������:�׽;ʮ�Pz�=��[K�Yi>�&��,��
F���v+<��1���&sh����=3� =܇���XG�y�<�t��Tĉ�����w�ּ81g=�%ƺ��O�Z�B=Bi�=�Gd=�G ��=��=������[v2�~Q%=�;�C���܄�CcD<�/Ӽ�ic<y�I�,�Q=��=��^�8��<\6ּn=A��=�o�=b1<hj�i�Z��=F=5ʽb*=�;�=��N�盢=�2˼�z(��7k�\
�I�=4��Y (�dg�<�DM=z��=T|��\��gM�<U�<'`2�����X��?���H�=�ZŻ˧)<�1f;�#���G<�t����<Ǖ�=6?a=C���KhR����<A��RP=2�<���Ȅ�<(AJ=���<&��=�9�<�K=:�)��=�*��Y4�-�:�L�L�慠;+]f��D�<Ctl�t��<����@1���'��k�:�#�|Zv=���=�x�=%z����ۑ�0$@�������=��2=$ӻor��d��
<���=�M�!�z��xX=���b*�=��c�h�9�n���� =��?�Wz�;u��&#�*�0���*=m+=�Y�;�R_=�*�=䱛��:=�d���>�� =�����#���(��+a��7D<Cإ��?=�6U:�S�<i>��A��Q���6�#�=;+�1*&=�xֽ�I��g;)=?��;�=#^9�{�b=��;�W˽�R>=,�>�X#�������;=Eq=�z켤�1�|H�(MW=Ȑûܿ���)ý<�n;��<�s�<p�7�u�&�_��E�=�M=��=a=�K�<���s��V�>VQ�=�)�=��<�<l�޼Y/��I�=�E�\&�����=���{��2�;=��=QRe�R>=kN½)½��<���;dk���M�<��=��ļ����w�=��o��=L��ν�/�;o��<Kܞ=jd!>��=���բӼJN=�>[[<��<�ۼʂ�x����(�@\����>��>�f���&���Ͻ�fŽ�jb=zH�;p͒=��=*�;Ђ=�O��8B<i��_P� 2�x �<MO%>۶�=*y�=�٥�ؽ�����)<�O�<D��<X�{<�L=r�=��<�̞���-<�9��yĽ.sh����׺|q;=Q���pa=�m�=����ou��h���닽�l:����q�(;_�z���꽚��=0 =&7ѻ��.���"���e�=C<#�2����<ȱE���!=I��y���+���->�D�������=�9O=���=�y�=��D=����볻=LEz����((���������z���P�J��T>�T=��<�� ��4=��Y=��<�`��Up<#���xd
=i/�<���;Ug�<�Ս��ه=k������<Q,�<|=���t��7;<.%e<�倽�l=��{����]�;��>���,�!�x=:?>ŢF>hJ�:Gڛ���<n	���3:+�<�<Z=��R<��<K��Ӡ�%~N�&2F>��ټS&��An��'�ļ]%��Ke�=󚅾Z9����=!��=dvh��"��y�A =o�㼎{j=����،�=�g輢:^=��ս�=
S�=r?=��<K�ڼ񰻽�[�=�q+>>ԗ�:��Z��=��1>���4�<�u���s�=ZŖ��t�<�/�=��=E	���=�"7���=�PF=�{��X#�=�,=�c4���-���;�G�=��:���<B5���q!�HD=�[;��:�%�ut=�4���z=��<�}m=����W}
�÷(�h಼�3=lW�;sV�<��<C���:�<j���_���w0=�о=0ݖ���;�]s�=%Ԋ<����z1��3=g�=�}@�Q�����=�Y=;IeI<ؽP��r*<�D��늼[g�<�Pb�͉;�\�=��Uò��P9=���"L�<�><�,��rg��Y�;L��)B������A=8k#>�ܜ=�ܓ<��K=�Aͽ/�=�v��w'�ex��d�]<�/B=Jz=�Շ=�A=`~½��}��!G=�`<Fu�=(�(=A���r\�=�R�=D�*��]��}*��d�$���Խe���_�9;=���=M�X<�~���	=�ު;�C=�7�<u�!���<�z���x>�=j@=�6�ӏq���6�Ȩ�=m��ܬ=���=ԯ!="2�<@X�Ӎ�=m��=�N=�U��B���^tB<Q,ɽ�½�ߓ<�	z=�(�zO���)�������4=¤ּC��=j�=!_i<���=�(����7>ν�K>m��=�z�<��ܻ�=� (��D>AҼ�c���4�=��<�p=�K����]�������W�<ѿ�<Iý���3>�u�;LȖ���<h|��q޼�D�=��-��h	=�F=�'缇썽���:�
�`�m����e5;c�=�6����<�5(<�v=����z}�<J��=:#ӽ��|;�o�=;��=���Y��<��3>b0>�f<y��sG�=$�<<��2�����?�����`f��k;�=�=�9�<�Q����P��%�)������L�=_�a=Q!�=��=�۽�-�������O<����e�[<z�H�M3b�_� =X����I�}n%=4%輛���d=�8%=Dd=T$߽���ũ=<�Ҽl�"�+�>%?�<���;f��=����)W=Gg=�&��o�=���L��8.�=`w��=.�;��Ш�<_���kw^����=J�%>�R�=E�u�́=Tr��9�=�P=@o<YT�=i�A�P*;�[2=�nC=��=�U��M�=�eo�C�Z<f�\<�>�<��<�����Y�<�P�=���Hૼ;�/��7��q$�wD���Z�=�4\=_��=a�}���<ϲ�;H��S��=ik�<M���O��i)�R�;;�,!=�`��V�=o,N=��j=��h�A���e�=�R�=��L<{u�ě=^of��ʑ��m�q��� �=]��=��������'>c��=5_;��*�(=�'��<��=��F��-�9,D*������76��&x�B��<�|��r	�:<��<g����=
u=��R<[ �<�z�=2x̼Z=G=犅��Ҋ����{�]� �E�>I�2=|��=78z�S�U=��\=l2�ݰ����<�n��[�s�����<�5��p8�õ������6C��X#��&��CZ��^�=R)�<Q�
>��=Q;�:~���W4=�
����!���Ľ@x<��="�w�Y��=Nzʼy=>�_����;`�9�w=��s�v��,Bc=H��=/�.��<� ������C�<�t����XB7=@�)=G��Ik=4-%;��K=ԘW�Ћ¼7d�<2�_�� ��=�G^���A�Cw>�-<��'�W�=���=q����{<�";&��J>+V�=!F�@�����_��*��7����*���d�����J<���=��<�4�zk�=��ɼ(���>
=7Kt����<�}�ü�
�=�<c�\&�=0%�;�`f=�h���;��ʦ�;hۤ=wm�+�ý���bν�,�=W9�e�=���y�w�Ğ��.t����F��=���<�]; ��:o��S����e=�>ϙ>���=#C���+<I�'����=��{=S��8 =���	��݈�<[�A=���=�Y�=rR>=��`�;�Ǽ���/Ϟ�/9^�*�=��=_ك�`�/�I�O�~�<7�m��M���M��9�e�S���!ẽ+h=V:������j ,=pϜ������r�<u��V�/=�!=F3��V�7=Pr�=EA��n�����-�������?�=}��=aW���1���ܻ�7g��}����<=�����=Z�<�r�=���=��=�;�=�˖�9F>�?=�k�=9C�=�C���#���%��fn�=��r��#>�b>��>��Ƽ�����ƽ�=�,Ѽ!b�=���==~,=F�<�a�W+�=&��a�=�I�f�$;�[�=�݈=�Qe�����I���_�=x�=O���AT>�W������1����J����k�=�5�=�1¼Ն�<�Q'���u<��<y��M|>��=�uݹ ��=�9>*���c�!=��r<��[�~i�<�N�p�= �c;/	�o==1�;�p �A�=�$�<7��4h���m�R�;-|G<ܐ�=�\����p�~Km�FE�;J�<f�>=5����s�<�CG=$D;��=L��JѼw7��:4�G��2 �~7����#��j�:���=E&��LO=燒<�(<��=;��h��<O�=���/;Y��IP��ʱ<���=��<��a=�����=�$=-ު���=b� �f�D�Ά<|�ļ��v�vݟ=�'<������� �<�V�=C���u��8_=x�<�Z�;�{ü��=�"�;vj8<h����x���I��O=�7����=��=�]���A>bb=�u=<n�=a�=59���.�<�G3<\������;��,��pͽ���<=o�<�3<�]��P:=�%+����:�dT=��o�	2�;f�P��ü(����᯽�M�;ޯ"��.�<{L=&�h�.�<+{=	�߽a�k=���N,=3��<��L=�޽N̛���$�B�<�6��^p��wT��<�P��z�;G�4�L! �τo=�WK<"S��2p; CW=��<9����ʼ~�/=M�ȼ<KE���<D�=���������"���f<2=&�!X7=�F��j=�Z=�`����]����<�<{]�[�;O����<���=P��=������<�f�W�<�r�< �U��RR=�X*=P�>��=VQ�<	�a<��ν���=+�=�X��2J=�LF=��F�=�M=�X@��ض;-�ӽv�>��^>�D�=�v�NR�<򣃼8�<���=7���4�<��l�/���6	�=o�ʼ�DX��^�0��=�1v���8��x����=R��fH�;�=�;U=(!�^:�a(=�8��DK=Ĭ�B*�))�=��=���=z6�;O8=�z<�+�<���8(C=�!�={�1����dI̻��s�w?�=���j�����1��&$==�=��ȼ�t���=�^%=�U=��	�E�=�ɲ�g6���c;�&�<�E���%��Y��J=i�	�]�ir���.��j]��D��=Z�}����<����W=��A�[�`Z����1=-��=�#�<gm+=��<2�>����&[������X8<໽�՚=��=zz�M���Bk�<�7ڽY��<4p�<ؕ�� ��&5>��޽h�'�-�/=����ٸ-��Q
�y�>�i�=�ُ��p���WH�%I��W�~=)J����;<y�!=��齹�bw;)�2=�I.��T����м����I>u=�N�=�~=�#>�����:$��Ӷ=��='��������?o=MY�;��<�%=�j�A�=���b��+�̽2�>��p2=�Î=К�=
$y=��k<� �;-�o<��S=3��>+=��;��Z=<>�;�=���\>�E]�=\k�;ب��.8G=�<�=G'��b3=	�=��?�� T�˟罛�'���﫽�0C��i�[��=`5W<>3�=r�ѽ4���<�IL=3HҼ�I������v�<�=W'<d��=���:"��=ֆм�#�<�~=?V��|G�Y@=�>I�W��^�����c<�8׼�]��g���@����=�)'��/=�Y��c�F] ��T�=߇�<^�L=4�F�ea=gW=Im�<�$�=ƙq=o�=�KR��7}��zA0=�j �Xe��K�=�)=s�=J7b�����_��U`���V������^�f?w�v�=^�_����]�<}��;J�~�s�V=�g�<,dڻL\�	,�=r��=��<�=��<f	=,y�<d��=[=k���R�=��J�z�q; z<���\���C<�H�c�==�%�1��='�<�����1;|�=�p?��wI�`�J�@�j/�ǹ4<[��=�Z�<_6��
1=�N	>���=ǯ=S^=$to<��k��E�^\��a#=���=#��=�<�T��!���<J��u┻�.�I)=�<��=�:<ȣ =T4��`�_�^<�P=�9g=B�<JU��>����W<jY�<��'<ʱ��� `�$�=�h=����`����=����T%=�c�<5Ă�{ww<j�=o�<��N��̄=u�O�$��i��|��<��:�B�����
��G=M���|"=�܍���\=c.��~�<$ʶ��Ȍ=�O�=� ��</'=I�=;�:'���-�{����҂�@�@;�k����p���<P�L=PN��<�;���q�=����]â=»\��g����<�g=�o�<����#&=&Ɣ��#=�
F=���=CN<F}�<W�s=�h�:��u��OA�ǅ��jH<v:�<X��b
ȼH�������鼠��=�~�=O�=F�#������ۈ���8�%����슛��3[�~��<]�=�0��7�=I��}��Ɍ��6l��s�ڽ��f= .v��*���=u =�8�=���$ʘ<\����.�o>=����%F	�� U=�p5=���~��=�锽�N��y&+���=�Q�K=��ֽX�>=���=�ov=N�=���<�@��s;/�=���;u*=��=B��<E`<�֯:��=Y҇���<��⽑=�ɣ=�q��%�����=`��=%%5�{N=��<ͧ<���<d-��œ��^�=i;`�8�5�z��<g�)>���=�Ğ�K,M;(N+=�P�\P9=u�=J9� �=
=�� >5�B�fᅽoߞ�2"A=�����>�&�.=m�=�.��`�R��H�=B�p�q��[�<O-=�-<V2h��ٽ<vх��ܼ�]2�\u@<�B$=ռ�<�f��馩���N�5����cI���l=�
)=����R P=S��=I�����H����V<�殽��.�Y]��8-�QnT�O����(	�ħν@9�<���p���%qC=��<1�;�r=��K��O=Ǩ=�`�;���V_;��&B<���<=eL=VL<�ȩf��L� ���\Q�[N��v�ռ�+=��3=��=�Y�=�o���-b�ژG=���=���<L2J=�E�; j��߳<�<=s&�=f��=S��C@ѽ� >� ����= �>�|��)�A���>��.�{K��E�k=�����&>��нx��<E�h={�`=@�m��= ���=1�j��0�=.��<��<��=*�=�E+>'�=ؠ>Y�w=+��;az�@%�����<�*�=�h����^f�6���g4Ƽ�L�o���A�<\��o��������=��<�+�<<����;����i</[L����<�[�=#�?����<�F<�<�䑽����9.�'�	=n�;��&W���=~=�%1;Н<��(���𻙔����7=�=�f��Irɽqf���Ώ�̢��E�����;���=-��<���=ab];��=S�<�f�<�*ɻc� W½p�q;=r�=]0�=~��<F�ؼ���:,*߽[ټ�i�=]��=�;�s�(=޲�=ͼႼnYw=w�ؽ`���6۽�;?���ټ��2<�uƻ�}��8�F>a-;=V�0:��==JVƽ,Ƽa�Q��K�������/k�*�ؼ�z=���;Ȥ<b�=:~=��;;�<�����e��A"�<�>�s���+�7���x�k=9�R�=������=ݹ#<��*���m=:f>�z��7=6��<z�	��q��������T<������;i<l���*ZX=�`����$꒽
��d�$�)}�<�;5=*�W= ������Y�z��h
�ؤ�����=�Z鼽�?<)�'�Qr�B"#=�������<풼�?�9	��w��=)]�=NT��S2
>r?�=�/��i�=	f=a4��ct��͢�py;<�7�=�*>�7���|��Z��hд����;.&<<�ʦ=m����� =��H�U��+��ڮ����=�f=��;J_�$��<��=���=(�w�u�">�<��dhk=�M�=7E�2���=�ޥ<��Z=�|�=<-�;��v���Ž��&��U;�(�<�&n���
=�d�=D	>Y��:�X�:(���ƼA�w����<�|O<��5��~-�������;ԹԼ�?<[ƈ���Ż�nP�djp�"Lɼ�Կ=��c��(�=�����t4��j=I.�=\9���bٻ?�ǽN�=a�=;/����!�.=g=?D�<􍅽����ܠ��5�ļ|$,=jc�x��M-L=q�=����O��<��=�j�;�Q��+�;��=� O=��=�v�;"ӽ��Z��xh��μ�K=Y�"�"���� |��M�=�~�<ty=��Q���&<�?�=���=� ;����<��[=w�<�L�<��=���=��V���=/^�=Aˎ=��sN���'�LvP�X,��E�=�I�����=�C9���H;"�������	�<�Cs�O�= ��=%Qǽ)���`P�<���;�rP<��������<���=��=W�=�Af=�_\<�O�D�1<�O>��D=����lե�&~��8��=im�<_f>=���<�� �e��=$�=Ԋ;�f=s��<	;�=�
u=��H=���;g�O�tI����}�}3:1�h=��b�'�#����%�R��bn�<�l�<�j�=�i��[=��=��H=�=����������2>�jݽ�;~ҝ=�h޽�=&>l=��L=��>e�T�닛��h>��ɀ�s0�r?���ټ�*�=	
.<s >�x躎��f�:V}=+�>��=�7u�ʧR�D{�;$���1�=�4�;�*!���=�=�=z�<>�>R�ɺ�q�;�3�=��{�9��<��=����Q2�ۂ��T�:�W%��������9e�=P白�@�`�=NP������=򴥽�'��۷!=;�"=d�h=t��]}��Q֡��5�<O�C%ѽ-[����=��\�/�=�9�M�P���GO�=�O"��#���	�;٧ػ�����=uv�;�E)��)�F�>�=�i��9>v}�<�%��ʽn��=�<�<l(a=4ھ�R�=���B�=����<���:�"��x-�=�!�<���<fA����=����O��T��=���߈���x����U;R �8vY=��]=.�Z�6���}l�=��=���#��<�j�xE*����<�>=X����t2���Ǽ��)�B��yC*>l�����h��c������'�SO޺�V��}�-;����3N����=6�K�!;�=X�=�$���Ѩ=���=k�:��=�k������<m��=�F�;�IԽ?�
��ͼ�<�<��r=V��=�W:�=���ٔ=|���m�n���s}=ʤ=�2!=�Ó��a���4=�v�<�I�=[�>;�=}��<n�o<�o;����C���(���,?�kP�=8$�=8h�����?����GC�_�;�H5.�9`Q�@1���
>QE�="�żާ'��5���b���
=V�M���=DN��	����=D">q7u��E6��$<��=u/ؽ�zJ����=z�����M�x�'�^�d��C:=c��=r3P�E$W��S���Cy�F2A=�>>3����#=�����=&)(�����1G���ڽ�4����*��Ͻwl=��=)�b���y��K=�����߽bG=z��9uP;�1�{{��1�c=�]
>ҏ�G��H�Ľ�;]�	iB���<�Z�=�G��H�y<5�V���=�'=%�L>�.��6P�" �<��Z<#�q=,+�==�=3����%�K9�;7�o�F�h�u�HX��&� �hض=h�$=�S��~غ������I/='�J��D<o�ӻ�ǵ�0-e�}��]���1����Bu=
�~�F�<)��=��~<9sH=�Sw�9�H��%-=*�=��?�0��R�<���=q�==��6���t� ������<�����<>�6�����ND�=0�:<�1ռx�����<F���J/;P}<�:,=���=_?�=7W_=ҋ�0dr��uZ<�2-=sH�>�>�SW�`n�=�`��A��Z�?<Nb���F����TB�=��N=qV��K��n�,�y��<��=h z�"��<N���W��v���������C9�<���=У�<�&Z����;Sn0<�≽�輊�>1��Q<R��=��<ϙ�=�jS=�u��+"ν*��{�>x����I��3Q�%/ǽ�L��sC���=�f(=8�V���5cy�	R���v=Tt5�߶N=���=�#��g'�<�Y�<��=�j�=���=y���������<N����=#��=\}<m	)=%!���\�"!��@�;�g��P�t= ����n=<�v=�/�%�8<�׺<��=@�����`����9�#8���Ͻbz�=��<[0{�g�=���=lq�X4��+>=K5ѽ��g�G���jU<��}-=���=�FU��~)��eT��� �@UX=Zt;"����K���ie���6<�==�1�=�3���ٺ=�B�<�'�=}r����/�_h�
��JC=p�˻���5࿻'�=_��֭��-��=3z�<ީ��ƃ���'�<{�
d�=�z=\�� ��;��=�^8��h���)<��8�񾞼D.��c�=��l�t�=ym��K'��z7>�Z=!��=ד�<!e�=�ON=L���5B<C���2:�9�1���_=�v���z<��@=��Խ�Y�;/�(Y=zeU��3����W(X=�n�=�Ŀ�Z�=��Z�������7=ŒQ=�>��<���nI�����`���uT�=��T��,p<e��Z5^>	���n*��!���W��&=�_���=�������센.Fн_p���=ڼk��3��;�M3�)F��B�<�@��o�f�į޽�]�=��h=d������i��m7���=���==���7�e� {�<�*B�#O��5t�=`��=x7���Ie=���=����0�@��3��=�5��aw�����=��'�����z*>����5=�T=8����=*�ͽ�L�=-��=�g�=n�=M6�<�7'��-7=�b	�Б ��=_-,=ʬ���_E=X�p��3�;�Ž>=�8��/���=���%3ؼ`��=����}{ռj�<<!ϑ=�(d�<z�<�h����7qW=�3���g-�h�����Y <{��aш�V�i�^"�=.j�=m�
�(��X��^1��$�����=G�=�'���Yk����=t�;/S���u��O��=��۽�#L�E}�=($�qkG=��
>�����=���<ұ=>��<N�_��1�^=5�	>F��=��A������5��J;=k��<H�>=' T�4sl��+X�꣘�^]�<��Ǽ�N5�.����V=z} �����Ѵ�<^b� �=.=�Y�n�f��[=����������������=�P>y��=�ʽ��|�A�w�A<�,�u���1d=� <X�ռ�UX���ƽIə�#$�=��$���RuR����fCk=q=�+b��W; �!I=O�x��)�>�G��fRR��f��E=�;�=ֺ�=�����͈=H���ٽ|�x��۬������U >FQ<�|���|��0�<ނ�qˊ=	bs��%
=�Լ��w=���;���Y�A=Ba��WY��zY�~����CF=-;�y6,�5��=�Y^<�!����<�C�<>�k�d��ި�<@Q��� =��S��-6��FU�v4a�箧<�	=<6:� �2>�J�9�#��X�=�]�vR��W��=��>�p�<s��=YĽD�)=E6X>V�ɽ�e8=�~h=L�=%�<`�K���z��}�=���w�����<5�߻������=0&=�����T��j����<����%�;�u�<���ޑ=!�<�W#��Sμ;E���E�<�<��N�<F�ӽ�y�<�/>%w�>�&�=�|����������k��U=���=�d�<)��⾷<(Y޻�-�c�<e<�+�=��X�o�=��<��=x���!p]�Rя=@�8�!��=&�g=�:������:��a/E>UT���R�=<{=��<K&��)Y�ޜ�4��8�w4�E��=\{=k굽�̵���<���<]��'_�� ���4{=bݫ=�u:���=M�������i�=�M�t�＠�������(=��M�/��=[l�� �<�,g�=0�� =��������1�;�A�=�$��T�<v�v<CXz�y��=N~�=��> S=��k�����꺊�F��=G�>�];�����1 =X�A�K�7=��I=l�=M�R=����_;�<�=�J@��$X=;u�=����|L�8N>��H��(�m"���PZ<�p�=�$���芽�}<=�M�\��;���=�Y�=�B[;�����r��ǎ=�ȓ; $�=�Rz������F�������=v�=>�s�9=���=�=��=w1�<��=�� =S�;�?�<,7A<�<R���VI=���=�m��2ҽ�!<�ս&=溟ZJ�ւ=43=i�#={�"�\��<Z�:���=1[�=s�s;s y�Vdx<�$�=MЮ�׀�=�@�=�Y�=3F=�|<Ot=S���ǽt�<���=d��=� �=�;��ߟ�����w������<>]ּEB<�C�����=)=��=)rػ����.�h�]��=����ר�<h���\�����p�_#�;���;�:��#j�< �r�R*�;��d���<Y�u�b*ɽI0���|���⇼J
��|O<"�۽�+z<@�P[�=���<�E��l=�g�=�>R��|��
ǉ�ؗ�=r�ϼn�A�i����fh�S�8>��)>���=;�{<�i��K.=��=b��]�<hf=���3�̼���=�#&����X��e���䟽�
�;LB8���a�q�=�7���=���;'h�[��:V�=���<�I3<{cv<C0�p9G�,h�k�������L��Y>��Y��j�;*�=l�=%Ѿ�ý9���������k=]`��;+]���=��<b�����~�X�,�7^ܽI�6�Bt]���=�W�<�cc=�i~���黑��=^��m`�<4����=lv��%���d=a����8<h�=o���f�ٻ�`����=>|=�>+��.�<=NԽ'�7�� h��9����4��\b=�˥�z���J���9���h����U=2T���_�qr��28
6StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_9/conv1dConv2DSStatefulPartitionedCall/mnist/fc_9/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:��������� �*
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
:���������� 2S
QStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_9/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_9/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_9/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOpConst*
_output_shapes
: *
dtype0*�
value�B� *��$���=:\�<���;
w�<<�
���2=?���t+�=�J`���>����S�]�*=�0�M{9�Qc���D̼��^�}5B<�`��� �<�&�i�<��5���μ��`8����&F���c���2��܄<2;
9StatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_9/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_9/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_9/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:���������� 2,
*StatefulPartitionedCall/mnist/fc_9/BiasAdd�
'StatefulPartitionedCall/mnist/fc_9/ReluRelu3StatefulPartitionedCall/mnist/fc_9/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2)
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
:���������� 27
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
:��������� �2R
PStatefulPartitionedCall/mnist/fc_10/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�a
7StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1Const*&
_output_shapes
:  *
dtype0*�`
value�`B�`  *�`|��=)��;ݦ<���EC�1,�<cݽ�T5=���0�x<�3�<���=7|C=D��=)������'=!?|=|��;�%�3S�H#<�?��4͒<#(������<$"s���=	|�7���鼷{��)��CT��Q���'>��==a<ٟ��+8�$X<�{����Rѽ;�Q<aX<;�5��(E�v��>�k"�y�ټx�½=^q<��<��Ž��T/��ԍ���w(���r=n��îɽ6Ӏ�7Gc=`�H����a�>�>M)ν�Oy�d1c��S�Ų�<�V���V=����+^�\J輋Ze�tʼ�0�=hT��Z�=��Ľ���de=?�^�����ͼ�y���C�������IK�4""��>���x����:�y�Q`�;�o�=:�N:= �]�F�#���,��Q%>�L<��6�u��=�.���^���(�Y�_=c�h�Y;%p�=�T��tw��e����(�/|��˧H���<��=�$�o<B�!�����o��<�U��9 F�)�
<}�,=��(=6��=GS�1<����^�ei�=m5���=��轠���۽#�>�P�=��M�ὤ<=���8���WM�>5��Y�<��Ǽ�d�=�� �m�M�,��=��=J�μ��]o�<o���o	B��
n<ę�F��=��=���K�L��(��6Z���M�<~杽w�� �=7�a��ݽ�T�=�j�=5����9������:(��=v6�<k��4�=��.�a�<#yR��q�Z)�;�>k"(���27=�~O��O�;�	��K������x�=ׄi������<�a��op���v#=w�=oV��sr��������`��O��lu���}+�oNE���?�݌><L�_�h�8�C�����23=p��;�.�R������P�=�%̺tt8�ڒ=���<�≽�0.=�Mr��;�=�<�1�=�9<=����=���ѱ�<�e"�_v��H��P"�:P��;����=Ӏ�=T�ĽW��٦�0�=E��=˶���,��Dr��Ժ[=��!��B����@a�:i��=e�o=8�=gh���=	=���<�F�=<�;�˩=�N���=>����#��Z���=�=lٗ;	���M�=c�<�k��{��;�a�=?{=��u�����j�㼃����s��9]�=�򁽿��+q=�};��+�&�$��y?�%����z���Z����ż,ʿ<�g�=<"?=��2t~��s=�.��w����=��8=�檽�"e�'~�=��n=Ļ1=�Ž�q(<�{�� �;��=��F=���<��;4�H<�(�������̽�<��
��'�<�+˼_��=k�v=�$<+=��>I�=\龽��̼�n6�Ʉ�=ǈ����9�Ѻ2j�uI=k�=�!�詼�y�='�����=7�&�!���g�=J�<t%3	�^z28��Ѽ�Gy�x0=��)���H�0��Xi�����-=��<�=��=��;,S`�T.��������<�Z���۽�j;�*xS�y],<u�=rz�/�=�VG�~�<��pZ�<���:)�<�%1�A�<���e<�k>t�5=+Z=�������;��y�l�;��ȓ=��
�tk;���=��p���Z��U����;��ͼ�=�*0=�rӼ�9˼�s=��:�Y&�5�=�F�=�8	=F���&=�ST��=��=�ћ��?<�r���J+=��=�/�v��l�<w�Ǽ_�9�ܦ�~�<d�+�$�&=]������\Q��X�Zӛ�ƞ!=���<�	.<�gQ� �=�@=�x!�bԙ���O=�@�<DFL=��=k�=W��;���^V=ʌ<�z����	ȼ�uD���<8��
H<bͫ<�O�����i�x}v<��-�[��	�<� ��Kܻ���2>��=���`~)���8g�����F;�v�<�wi���=*���n�ؽ}��E.��쟽{)�=`��<�w߽D�-�2~�=Rw����{�=j�=��e�=}� �gQS��q�Q�=ehp<8Q�<���f�M>6�7��޽d�<bս�T��-��j;Ai,��ƽƊ�����A��<_޺������=�ؼ���j�p�-P���żQ���_K��m*�7��͢���G���>�h�</�R<y�r�m�p��q7��u=W��<�/�;H���)�=�
<;�;�׺�I|�2�
>�9��)!=n/��`S�0S=>e�=��W�����fX�U�����Ӽ�؛=�Ce��tݽlԚ������+�r�g��mJ=X3=-=�<J}<@�ϻ�v�<�=�e=�<%�	>׏(�l�<�����R=Qc�<o7�=�[=f�d=�n��g�e����<���<�oƽ�k�c%�<2�k�x��a��xb�8#>-��=��|=o :p��=����=ٙ�C���VF=���<fX<TC��6M�@F=�6�X)d=Y�^�c�ѽ1��#м���;5�< ��8/�=���<i_<��2�U:$i��U����}�����-�k�=[G��s���� �o|(���=?�l��传�'�Z��=H/==�>P$�޻;��% =D��=�<�G��=��Խ*Q]=K�(<�[����r���?��==9�	��D�=i4�*��==Ǽ��=b���Φ=/í=�=%hD����;v����rl=��U��)� b��KԼf�t�|W�Q6ͻ�tн��<���k~�㨼�u�=��Y�\�<��������{'�=��=��>��:�:�<򿥽�N�fT�<�����l�iy�:�1��>��<����/=U)�;���= ͼ���;uЂ�����	J=�l<=�<�=�}�;��7� ���<̝ý�����Fؽ�x�)و��C���$=u�����\<��5�;+@4�/�=1���as=�hk��a��DӺ=p|�=	�bK�=+&>�>�<,#�=Dρ=��=5������?o�=b �x�;�R�>r�� �	���߼L�Ǽ������=�_=t踽%�����H~�=� �=zA�:��<~��<��V�o�pj���V<WM2��2�<���;F�W�s =�R'=�&�l�+���x� �E�b%�7�=qǣ�������� ��>��.�S<Z�<����w�k�<�����q�=��9�IJ=��=��&=��@�߱6���0:�S=�G= Q���"<��p�>C�ɴ=��}��ɬ<ּ,�=A�����<m���Fc�=¬���樽�c��%��A)�;����ꚽ�#��5N<Yǃ��@�<�ȼ�s;�����a�<�<���=؉$��Y<������	���:����۽mt���=F�Ľ}�=���Ӊ�=���=M}G�N��<`)R=7u&<uf��!�Ѽ�
���/=&#�� ��<?�^=�|�=�+�=�G~=^��[���4��6�=�E=j�'=��<փ*=��O<�ǲ=*�<t�C=j&�=k��=�R=�7��7�=R������6��p��j�<��r��3�=�1�f1�=�2(��=Ҷ��۽~s	;��|�3��;�&#>,><�:G�_|Ǽi�=\V𽣙���ҽ;w�=yqH�1 	<|��-n">Pﶼ#���Y�yR=�Y���=|n켇kk=5ʥ�u_����^=��"�eμl������_=Z!�p�<����~��<vn�<9%���	V<��<E(�=2�=rZ"=8���9�<N�a=�7<\��Q9�4ǘ����Ȅ=�mͽd�w=q\���!��]���=}�=�˺=7
=5�½���=�s�#�=s芽f�=��:`*�<R���(��<g�=LI=8Hý��=О�������=�V�=�B�<]� :*Aֽ��!=Fʽζz;��ܽw�Y<p#��!/�Yz��{�=��=<μ=RVT>�]s��aM<��!�觽A>=�?Ƚ��J���|<��ɚm�?�O�Y}���;�<��^<p�h�qA���<1N���<�T
��5�<�I�=H�=}�=�V=6���p�ҼgΎ<�DĽH.��C�<�E��=�Y=~��`�#����=8��N�U��>��U�A�X��;�=����Zy=f"=���0����6�%? =�k�*�r=���=W��:t����M�OK�=�°:�G��$�����p:>�x���~�D�� 
>o�ڼ�̐7��˽])�tL��XA2�<Sļ�(^��-��j$0:L]ý�]��%�`��x��X=�P0=�=�;�\ڽ����h�.=��7��+D����T�x��`<�<����)�=�	�<��8<᾽�7���1=2�=!��H�H>��Խ��|��e��ގ�342�������c��P%�2|���\4�����ˀ;Նڽ�7��*v�Ǜ0=�����}[=4N>΍���Ľ-g�����<�re=�T��q�)��uA�z�=�)�
�Ӻ���e�b= o��0�<m��?�����|����.�tZ����<=���<U&��<CV��1ɽ���֠0=C��Y>H;N<ǌ�=�r<����6=�N��ҽJKM�+�=6��=�J=%=�a8ܽK�<:�A=� ۼ�]���Z<�o;37�=;`��ꂂ;nM=��;���<m�:M5Ҽ�=s�"=���g���aס���ͽ���=qܖ<u�v=4u`=Y�m��'�D��� �)=�Iy����="d;�d=��q=���=�Y��n9�.��<AG=��8=��ֽ��+=K�=p����W���<�|<=r<>��=Ac�; �)=��!=UP�<�Ͳ=��!=�<ף<=��=�+=���<�%��=T#���L=&zT=�0�<�~�<�^��XX�qȶ;�&��U'=f�=h��<L�<���B�	�z�ݎ�=�=zd=Q���<1=.z�={*K��~�6�(<���Q�=�D��:Y
�b����\3�mʀ=HX�<�
o<Vb�;�U%��!�Az8=Sd�<l��u�;��<M�=mA=��_=�r�<�CM�x#C<^hO<3�Z;�)��#�=�3 :����t�5�6���8r�<^����\<64<6�M�4�����<,�=�_<C
�����&�<��;c	����<���t�y<1��=x1�?;"���=�S��V!������=iq�< �1�"ʟ�������Gƶ=&�\�*����}w��	=�Lܼ�=��T<FB9r��=7��=g��=��
=���=�(M��o��[�<��	�Ў�=ܼ=~>��D��<_Sý��%=�ֽ$R�71�d(<5BO���V�Ž��=4ڂ<��>xҘ���k�Q|����u�g|�<���J�-=VA�^Sj�d�=1���3=ɐ�<�>�;Uc�;M�˽+�X=|��=4�ͽ����Ό�[�^=�� �i$=��_���>�/"�o�뻶�G=Y<G��<������\6��}��n��<u�<��=@�{� ��<!��熌��f�<���=X���Ԃ�w�@���ݼS��<�@=�7�<RQC<f"�<�Ep<s�9�G�;���<��m=�F�	��=�k6�v�=��5ؽPӼ��j��>���(�<:�=S��=9���CH=5r8����Ɵ�=3�:��C�<&���xV��
��<�OS=�3<}�c�=�k(">-�W;���<j��=,X�;�=��<�2ｨR�=��4=f|��?=_��<�ǒ=@��<��=Ŷ�=�|I������'=%ս��ʽ'�3<��*=��A�C��uߐ<FyD�
�ѻ\A4<�F�;G@N�_\;=eTϻp���]¸���ֻ��z�y�,=dG =$f=BX�<�i��"����<:�=G�<�8Ҽ������q::�<w��<;�6�Nv�J7����<R䯻�8�=���36)=vw==ѧ=�E�=L�;IR�=�T�5�Խ���=g�G�+���֒�;�=�������܊=��?=ww��t�6���޽t���럕���L�\_���߼{0�j�������|=ۊ�w�<�����F=B������<�t�� �<T�+�<L��:���i���A��hf�������j<#�=Okż� �hf��8r+���:�)"���G��=��*>�w��bཨ39�6�=xĺ���Y{����<�����_����=�R,=	��<m� <�{��˘��.��зV=��򼤴1�#�9�˼��Wl=��k�ӌ�<vQ��P<<��z<e�x=Z��=�WL=��ûp�<��+�N-��6�=ᛝ=d�?��J�<��<?�;������<sq��;{����h=�����H���Ћ�Mb5�������;�\m<l�����⼛�ƻt2�UXu=@G��C��G���ک߻��㼊�`�3�];�~��j���X�/;�=O4s=�ft=mt=7����)`=P���� =����O�<�j����Ѳ,�
��<��½�缽�1����p�+��(
<��=��n=B��=�>��29;n����k=.M=}-=�o6;p������(훽pݼ|!����Ͻ�=n<Ò������, ��cѼ����~�=u��>�w4=�4�n��Xxͽ��z�#���5E=�ƭ�K@�=���=
s�9��A=���86�(�?=3#=�˼�M��ඖ�|�u=L��7�S�=�4�<�޻W=������<�|�=|.�����f�v:�ȕؼ������'=�����d�<�!��j)�DG�=�х���m<2'�=0����9���=��<��셰�Y�<�`��+�D=���5=<H�p=�%���Q<��<�0��=��=/y�<��ʻ3���������v<�kʻ�C=� >.0?������l��BS=7�=ƶ�{:a;� ��������=�D�:��=!�Ｅ~�=�	J=��l<���=�!����l�~W�S��h���d�
;�%R�� ��� ;��; �|<�!%=h@\�Lr=�(����<��z5����\<�D�<=��z��&l�&;� <�=�X�=kW=> �<�NG�l�9E��dۦ��}ּ^��!�=���=��=���< p���o$��N	�-(���廽~	�=s����;;ԗ����\���t=��!=y��<��o=���(���6�="��<�;��1x#��>���t��a�=���=K=��^���;c�<�:?3�㤘�O�=��)=¾E�e�:l(��{�=o��wW=��h=$c �$�=[85:ZIj=>�=R�<�
��1=�JI=�/7=�FS��O��L�=���<7}�:�=���<0�����ļ�$�<�b�
an������F=�$��<�SN<��(�c<F��&��;Ό;�3�<p8t<J@�<����+=�똼-#!=$�c>��z<v�R�tL����˽)>��x���>=\��|�N=��<	�u��p����ۼͶs���ƻ(eM=���;eZ2=O�=ҹT�����= �l��<���*����N�.!�@}�к<��)���P�7���Q9=��)<^�;_�*�y���C:6���w=Jͨ���+����=��;<eO3<�x�=�h:��=�I���cH<t��M=�؎�^&D<
�<�U�;^=�P�=�r(=y�p=,?��ͪ<��<��c*:<�̘=#�h����S��-B;���1>��'a��?�S��ޑ�<�]��W߽��=}Xm<�l����мL�B��~��=�wf=�n=���=e�=I���Y/];�U���nZ�)�<�]��dz"<�}�<ټ=��=y��:��-�0����+���=
_�W��b�5�$��<�ZN�b���c>�)#=�7߻��<H\�<�7Y����;6ϸ�S�꺽����ܿ����/���;�:�>�<]=�Yl=�{��g=�,ܼ밭�_N2=ego���ǽD���	�=j`�<n#���"�\����1q<դ�<�� <?�C;*z�=egX��/1�X3��v���|�~{�<d�,��1=��Q��>��\�<'�잚�*�d��!��^�ޫ�����oqw�����M��<k<$�Bͬ��e���Wֽ����B$����;E~��`	'=�*����h��<c�<��`>�ˣ;B��8w">s���5=��Ƚ3���쭚=�H��t]<)e�T�=��a�I�=��8��Y尽���=��";4�@=����Z���j���#𽼨U=�e��#%6<e�����H�1���C�R9U�˼�D�����L�;yt��[���м������<�m
>J�<��#=.˽M�=�_Y��u�37�kݜ<u��<�v�A=E����W��<H=���M0`��*=M�@�f5��Q3<��K���O�=Z�ƽ�Z̼�t�<FD콧F=�B����<�2ݻ	��#B,�ϖ�=p���J�=c��U���S=W塚Y���[���ݦ=���<7aڼ��Ҽ��=�_49�N�;�<k�h
�w	��B6���O�=�ʼ)$J���>�����<���;�E
�����s<S�����=O������<%����c�ش��!�<��0��A���z˽�n��蘽Ȣ۽��;Mi���t�� ��=���&�</ݬ��]K���>�uƽ�,<f�>!����=
��(�m�_��3}���9��,0<��Z4�����<料��(���?~��S�=NI	������q'�_L=1-(�>	��m�<ݎ���@>7�=�=J۽sKf����;��G=�i?�:���6�<�*<��<8����A<�L=�;�=�F+=O��=���=�䀼F&)���=a���3�F=L�
<�>>)	Ӽ�]�<#�I=ˀ{������;x �=ۍ��;�\����;
%�;"ln= �|=��W<���=b�=R��1���'鮽�0�<����$>3���2����ݼ��4�����5�{�h�,�H��X����0�ͫ�;Ybt�����"��mɽ�AսW���3D���:>ҽ*2�=y ��Žuq��:�:��(��ʧ���½?�ܼQ�=p0�<Z~��G���i�= A���*�N��8�=�љ�cڿ�����	�G��"�<F�,<��/=�0m=�Ᵹ�S=[\=���Į����w�=q�=æ; %�=E�<Nx�2�̼W��ʮ}=��L��&{=�ㅽ�e:^�<N�����j<���;ᙰ��О��@���%�i����.<Y����[��돽�8���Y@�w�<��%�D�-�DzU���5=0~J:𭂽q,�=���<�ݽ�Hw��>J�^v?�Y^t=P�?�9������=
Q����:��ܦ��y�=|���騄<uZ1<����N t<m�мtO�����=�
=�=m��p0��!rU��� �R��UC=��=�RE<L�)�4�<% ���[
�M]���>�z��<��=Ϳ�<��<��˽���n�<�3�<�(	="=Z��=�S=b�=��!�mt�<��L;���=ΆR=�ڜ<z^�P��=��0�z��I��^YE�`�4=���:��> ��<�����X=�c�P�ټM�L<�pX<"Sb=���p���U�W;PQ�=|�=h5�=�b����!����=���Χ=�u1�E�<uR=��2<�E&�*��2KG=�<2�B=� �< ���,�;L7��,��<��^=��V��B�=ǝ��p�:=��9���K�ϻ�����Ȏj���=·�`GI=�Š<l�W�n=�5=Y/f;��=��0��̻�Z<.|�=N��<vh"=��i�B��<
�4��2����=�h�ݼC�=%r#�l��=љ�<�hw=a�U��g=
�p�������>�I�1>�`����<}ߧ=򄬼��[<e���n�2=��	>� ���
���0�A�B��==��4�߽��>�!�<�8����<0�<������=j;�x�H<�侼���=��a=?��m=������ͻ
�<ႚ<�n>����(�=�l���	f=A�ɼg4L�U�-9��ٽ���<�%��K�<';�tZ�=�Y[=i�<�c�<w=-�<�&=G�a�;��=�V�=��=�H<��W< ����<�_�<�E�$W�������X���s=�!�A�戴��v��/�d=Zӂ�%M���H��^�"��ԽeB�����?��������z8
>C6e�Uy�<�ژ��4̽�Ļ����>Oo:{~ż\>�H��-���]�A=ߌ��oB���H����Jټ����4��;D�#=Sv�=�x2�hA�k�߽���<&2�=������<K�*=ۉ=� �=b<<� W�1�k�r?�Ty�<Y2�=�<=��A�ݟ�bl��⛼o黓'��n��y��=x�м��=;ƭ�GX=����:��M��}ƺ=����8������7�=���케�>=�r=X�<p�:+��w�b�gì�����<�<r8[=�2�*���U	�<,�L=V�$E���DO��<w=
��<"A�<Fȹ=_�4=�w�=�v��[	�=��<�4f����m�>/=��J��;=�I<ӕ=��ý�9��#�x\����F=x��j��<�X���7����=�`��1 (�����m�=g<�=���:z>S����<�I껴4�<�����h�=��=����_���}8�C���D�۽G)��©���iN=�R����'� =8�T���U�����鶼��޼"_���k�<�k��'%���<����\�><$������`"��Z={��;!/�=טf�A0&��C¼� =ڎ�=�hg��τ=
�;���|N��Eʅ=j&>���=�T��{��r��<�a=S�}=��Q��z�U"�<,�U��W"��|:*�<��-<��e=])>�b;��l���� =;��<n���B��=M�
=�*i� "�"��<0�<>Ձ�R�<r�r���=��>�_�qX�.�=�����qʽ�� <&q{<e�2�6X��[��!($<����{�=C։=bAĽl}�� �=�|>=�
k=,�M<�?>�=t�I��Q�L=����=�g�,��o�=yŹ=Џy<�	=�P�� A�3k=�?����<�砽�:"�7����PI���d�P�=z޽������'A���3�i1���4=��J;�r=��z��Ӂ;s��=K�=��<~3�=�c��RY=������<��=ؗ=����)�ｿ�<�>=�5�<ҽ��rc��ۜ���Y���	;C�Žf�4�Y�U�����X=`��<b׻��7��Lּ�~��[_�#����{�<�������=)ͽ��`�x�=I���;�=h=�y�����a�h�;^�<<�J��W�K�u��ӻN穽���=W�8=���=����S�)>�=e�8=�`¼"�<$�=
7�<��+�S��s8�<�I�=�vp=7�S���j�=�
=��5���	�q�;�s=�潢��;/�=���=f!����/�=~��̼oc���=i缼���d=n
$�oB(=�#�<�1(���н�L=q�<i\�����'�S�	I���H�;P:==��<�	<�#�=8;aՅ;E,A��F=.�=�L�=����ǪE�Rx=�;��\,=�Ź�
��<O�.=�x伺<�A6&�k��=������<!�"���<���;W��=^�=t����`�&�=��=��=D �<3=R���%�����=�i����(==�
>Ҭ<�7=����L���X��z��a�ʚ>�7B�5j�b�{=y&�;��ʽ�K=Ad�;��#���R=����$I�����<<���Wֽ����;@̿�8��=�QO��Ca�2G0<�����'�!m=�<0뛼%ڄ��,=��<W߽�����E�=_��<)�I<)Q<���5��<����u�=��;��='��=5�Ѽ�=�<�a7=���@{�=��e����=m�<��ս��=W��<5��씏=u�=ҴN=�%��(*�� ����Z=�zp=���=u��ʋ�pƾ�u:���_�;�w���O�=H0,=B�Ƚ�%�<�/F�h��?C꼊8i��2�΅�2c>�'��A�3=���<��[��=����=�!��)<*�˽�U׽�i���e;��.�8#��fPX=�R��ʨ��#�<I$=7q��_j1��/c<�i�;bAe�1c����=�Ž2��-T���}��0��6�(��*Ҽ�_���Q=j�½29
7StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1�
*StatefulPartitionedCall/mnist/fc_10/conv1dConv2DTStatefulPartitionedCall/mnist/fc_10/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0@StatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:��������� �*
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
:���������� 2T
RStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
2StatefulPartitionedCall/mnist/fc_10/conv1d/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_10/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������24
2StatefulPartitionedCall/mnist/fc_10/conv1d/Squeeze�
:StatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOpConst*
_output_shapes
: *
dtype0*�
value�B� *�"��?��<�Q���U�EMK����e��x�=1�;�򥼬㼈-�� �z˼���!�꙼j]��woּ
����ü�m�;0<ڌs��
=�HϻHM2�di �1��4���l�2<
:StatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOp�
+StatefulPartitionedCall/mnist/fc_10/BiasAddBiasAdd;StatefulPartitionedCall/mnist/fc_10/conv1d/Squeeze:output:0CStatefulPartitionedCall/mnist/fc_10/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:���������� 2-
+StatefulPartitionedCall/mnist/fc_10/BiasAdd�
(StatefulPartitionedCall/mnist/fc_10/ReluRelu4StatefulPartitionedCall/mnist/fc_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2*
(StatefulPartitionedCall/mnist/fc_10/Relu�
,StatefulPartitionedCall/mnist/fc_11/IdentityIdentity6StatefulPartitionedCall/mnist/fc_10/Relu:activations:0*
T0*,
_output_shapes
:���������� 2.
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
:���������� 20
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
:��������� �2S
QStatefulPartitionedCall/mnist/fc_12/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
+StatefulPartitionedCall/mnist/fc_12/MaxPoolMaxPoolUStatefulPartitionedCall/mnist/fc_12/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*/
_output_shapes
:��������� ^*
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
:���������^ 2U
SStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
+StatefulPartitionedCall/mnist/fc_12/SqueezeSqueezeWStatefulPartitionedCall/mnist/fc_12/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*+
_output_shapes
:���������^ *
squeeze_dims
2-
+StatefulPartitionedCall/mnist/fc_12/Squeeze�
(StatefulPartitionedCall/mnist/fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2*
(StatefulPartitionedCall/mnist/fc13/Const�
*StatefulPartitionedCall/mnist/fc13/ReshapeReshape4StatefulPartitionedCall/mnist/fc_12/Squeeze:output:01StatefulPartitionedCall/mnist/fc13/Const:output:0*
T0*(
_output_shapes
:����������2,
*StatefulPartitionedCall/mnist/fc13/Reshapeӭ
:StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOpConst*
_output_shapes
:	�
*
dtype0*��
value��B��	�
*��b'���@=:�=�|7<G|�e�<띂��v}���8<��	���<4�F<q���J��<��9=v�==L��B߼*nr��z)<�=��(����<�W�����|�}��.=Z_�<4$����<R�����=k9ۼ�#��e�<v��<��2=� <񲘼=0���y.=�@6<�r��yc�x���:�\��eW;��i����ɻ俷���;p6�����L��<긽9<�=�Uv=��켸��<�;7=�	��C8=TVs��p����Y��X-=/6<{�;�Q��<�e�< �;�yf��g/��g�<����$�;��@%����c�-=��7���3=<�u�6<0�<�/��7&���;�<� ��>]<Oo�=Tsֻ�K�=n�@��4/=	V��j��������<Prc;G\�<Ҩ<f�+���	=O���Ö<���<��h<�s���<���:��&=-X��ʓܼ(����i�<8Z�;�}�!������<�M ��I�:�r�􆽯�:��[;��
=�'>��9�<hX>���輕"�<|�<k/��	�=��w��V�:����6�@xk<���<!�u���=ogH;��=��M=�H¼g^q<�r�4�y<��1��<��S��r#��P���"�2��X���"��DE��Ѽ���8��՘=��<(�������	��g��:=1E���=���<��=�<ϤF��<<O>�<ë
���=<P{;�&0�ְ��a-���٪7<#�J<��ϼ3�<2�=�7������R=V ��Ŵ-=���<ӅW=l���e<'�='6����m����=�y����<q��=tT=�v�j���B�=�b���9�>=�ӣ�N,�<U8!=ނ�=�K=�������P�;R��bd�=e>G�ҁ=�==P4���<�b<L��:
a<t�V8�<�˼a&e==��<��$����"s׼���kX�<��=
\=q`,�F��<LK�;�s��H?�a��<7����T�f<%�Iӌ;��<+j:��� =j�{��/E��
W�T��=��;�#���k�<�jw=,5����;��a��E=�Z�<9���<�഼eE�t;)�9��1;�DS��on�<����<}�<��6�au�<#�b�}�F�=U�1=d!��@#�x1
��;�;6��<`�ͼ�y���̼2r�<%	 �`,��6<�5ӻR�'=~.�*�<� >�����u=ֵ�<V�_;�������"�]�k=	t*���=�3����=ԗH=c>�<8G"=^���29q�0�<?֜�Zԣ��NF=6 M;𥉉���L;}/�<N��<�e<�+-�����W�<�g�<�ǚ�1��S��j�<
�@�7�N=�ͼ�{$=����I�=�=��J=	E߼��G��][�<�|�<=���a���=����<N�<0 �Q�K;�e���R=��̼��R<�<�Se��Ƌ<J���?<�냽y�B=ѣ������ =�>T���<,l�=)�a<�r�rt��C�ϼ5��;{�<�jü����o�\�;�yb���ټ��n����2�}�=vv=Ծ�<����K�<x�B�I*=aj=��v���<6`����7��;�u'�_�<#�!=&���D޼,�(��VQ<�aW�=����j�<w��<�7f��!�<l�-���+=��=�u+=^!�+��;fM��Q'�]� �L4�(^_�ͷ
=˰<�Z<���<�k��pJb���(=q<2=LB��XW�p��=���;�D��qYK=��G��H=�&9;NL<��G=�^9��D=:�ؼ;'�J �����n8�Y�Y=qT��Y=���������0�CN����P��=��>K��<�Q��
���m&=�):�λ<�7�<��Y<J���.��^�򼒳&�od=<H��e�<;	�;:tͼ3z�<�����#;$��<4���Ӛ�<�.<���� ����Vg=h�(�P
λ��=���@�b=_�|;�s#��6:K��<S~ɼ�Ƽ	`�����<��=0 �;�<�����r@=_mB����;�T=Y�ǽ6!0=�y��h =���������T=��>���{��l`=��Ǆ>�?�=��,=��=�d�:�6�<�/ǻˋ�;{d��x�'�OU�=j��xW�=��<�v���KQ�!sS��G��9��}�L����=r�U���ɧ�=⹃��N鼬�<�	��<;��;�R�8;���p7<�x=�Y�<�`�;��l�g[�9ucɼ���<���=.5���`�����y伾��M79=��ּp�=~��<6��V�;�=1�	=�����<^,F;�U��$��2��J�ݼ�zL��|�^ ��S���؈;b�G��T�<J��-��ZO<�ںCH�GѼ��W;�0n����=
�（�����h<3���D��= �,<C�;JL��*��<���I�RR��
=P�m;^� ��C'�����L_μ�����!=s%<f��<B�-=j�"<0�m���,=j8��&D<�d��;���=j^x���==�(�7E�<�Í=�6�<yN=} =���<�8B�����i�!�껺< �X<D��<@K��Q�=�-=dn���Q��_޼ܘ����@��Ks;3�����O��=���m�� �<?
8��f��.q=��a�ciN���.���=���;�����
�r��D=�)B=o�ؚ:�Z�W�������<�y�;��2��X�<��<�m��+�A���<J6<Y+�|����S�`ɨ��w:��{?=���=�X�9
�f<6����<p*���Z$=���H�e�
=3.S��k;�F��|-�>�<�F�<B@/=C�z��O,�̗�<��&��c$='/<���;g���3�^=��Z<_	�c��d�`���:�SՅ��>D�r/��Ś)�%JH���"�\�{��^�Ɖ�=�N=����R�J;��u����;����<c>��<P<�������Td<(n�<�41��l��\EJ��~�<�<�z�H��·��hH�� ����0����G��@Y���<�Yt�ҜD=�qv=��k=j3g<zQ\<C��<��v<|�%��D�< ���؇:<��<Y�ۼ���z=�ZF��r[�`�*�.�T�ux�wHK=�e<��=�gD��$�3��<�L<����ɼ��`Y�<Ȑ,=H���m��=&g�<��ι8S��v�=<<��<��:<��=�+C;�St=�"\��la�G`��:�;,�>����W�����Y�Â=�Kl�y�=F�@�R����<X�4=`[���,
=��9�(�=m�ͽ�mb��eS��e��Jϕ=$?N�d�Q=4����
�H=�^)���<�߷������=J�3<e=}_5���j��=�j]�ꗌ�$�L<�Խ �s<���QM=	����=pÀ�E�A��8������lY4����d��;���l��to3=W[\=X�ռ��G�m��=d���{���<��������oLO�ct"�?��K���"������j�=�� ;�l��1L< ��=g�ڻ���<򒂼�A�����Ɯ�
�<sy���E��	PB��B=�0̼]"��<���+=��h=)��;�����7�<�ͽ�֯/��<2�μ	��=�I���퍽��T��W�<���<2=~~��ki<=��D�/��=���<ߋ=����i�����Ѽ�u�<�=��8�9<��� a%<�����$=�~��������<�|�<q}U<�S�ѩ!=W��;K�λ?���kA��mP���3����<������=>ȟ<�*�=�@�;	�M=��9�3|�=�S��4>����⼦�ý����,���4N����n�k\�~�T=|����2p=�~�!��J+�<�2w=(��<�⁽(�o����<E�=�?�;eU��׶F��ʍ���=�AO;�&��"���W�»�=<W�O��/�������<#�r=rf1�ݝ#�{�$�ZAw=��4=s�p=:�T�:*�<�\�<��L��e���X�T4��x���7��>�!g\������[�x�;'@���B�\߫��Q����<P����=X	=�g=K =!'��c��/V���Y��t0�|X���<�5� ����ռތ�< �C����;�o=b����D��A�<��ƹ<�>W�D��7=�vi�a.�m�i�����R��w��Zٽ\�N=M��<���������<)�<[��<$=�<�.��#�1����zY��|�U1>�n������<�λI����)?<�_��}���<���:7�5�)-<'�~ ��V�-���>�gֽ�<�ݼ�4�<S`����g=b'G��`��9`�!Vy<CC��E�<H7��Q�oQ~:Xd�;���<\���=L��7i���*�ʬ=����A=�b;���@}<9!=��"��#0=T�~�š)=���<�~����<0�<��=�1��r̽��ƽ�2�����<3ʟ����vʱ�'Kc��8���f��G=�˭��6]�� >cx!�����`�H�5���*��=��|���E��M��ry2�A�=<Ѻ�<7N�<9z�;��<�2��<f�;�j���%.<��R�Q#�5������=�5=�fO=3�<�狽�3_�A�۽�x�;�)��/Z��)��H�.�Zd�5-E��Dq�_1�/Uڽ��=%�=��J%��o�<'���������`�=r�>��=�1��LK�M :<pO企���j�&���=JÆ�B�=�<6=�4�;nּ&P�=�ǚ;���<Y��M�1����~�7=���:��������'�%=�ɼ~N�0�����~�,Z��, =��4=��7��C<N������8.�;�4�;ОѼ�K��E��;�s��f�<��'<Q�=j� <p�@��a
=��h�=I��;��n9j1��5�gg��=$s���r���<&Y�)��=�T=�C�<���P>��F���;���1V&��Y��yU-���`��:)�����<?G����;h{=�橼��,= E==dy�Bm<tk7��F=5��񇆼���:cd�<�b<�4;���<PJ��X?��*���sx��<�<a���Q;=���=%\�:o������;Y�=�L*����=ҢY���Z���V<�q�y*�;wռFj�<i;E���L&��2r�����	����RP=��üo��;k�y�~=��;�%�ü1�� �摔��Œ:��g�.�$w�[z=�^�<t�~=�r��ST���|
���;I$�<���f������N�|��K'���s�̓߼����:ޣ<�����<�<�y�<fIv�d*��0=ֈ�O6)=�����Ӽ���<�� ;f�� �><�tx�8��;�\�R0�<��;�(=�o`�����6�=*<��{=ٯ��9�|J3�����=�^�M�x�ȝk�m�9�R�<>n��&<�! �<@V�.I�<���<��3<�B��1=�N�<�_�<Aժ;"3��l������<�_=�I;^�C��"鼯���'� =(]�;��$����<�=zy;92�;�?ݼ#�1=�M�<,�^�dHL�'�O��z^�RG=�<+ea=k�#���V=zλ6���ֲ��3½�ŭ��I�=u���Œ������eO����
=���n����f��x��rh��y�|<3L��!ż�R�<���W<�$����M|#��� ��;�G#��[=EV�<FS�=����\��'ؽ��7=ɇ,�T�4������<������j� 0��:���` ��'�p<�⪼9��h��3 �e�;4���4���c��Dɽ[H�=���;��.��c�F��%�G=8���F�,���
��x����;�?G�d�q=�\ͼB�(=��=,p�{$��g������"�x=�����<�g�<Q������s��?Þ�D���Di˽֍-=�?����<��$� ��<��C�0��=d�ӽ�C������>���9>=����Z��=��<�݋<�<�M?�Q�L��ߍ���D�}�i����{�#��=�!��Ҽ�����d�۰=�aN�+m�Խp8=���;���<����E��p�9�q<B�?�tͺb5�<�<N5�<�թ�a�+����_~<l�;��漢b�<�o�����=S�v��핼���`E�����=K��<��<t���_,=5(^���)<�_�� S��<?9����H=�j=��<e���
<@'�8�n<B��<�
=�T6�e�-=ݜ=?�=�d̼~b=z�ݼI��<�PY�^"R��.�<g��< E%�5zU�s񏽤7����<&�K�4��ǀ�N�M�O9=��1=.�</}A�_&=�)/����=��c�ZE�׉V��υ<ӄ��~��<tx���߽l�ü�B�=� �<�N;�^m�B�<Nⅼ�Ľh��<Y'��3�H��=d��Iv��hܹ�H��T��}�Z�����6�<|7@���=6��<E�4���%�|=�/��#�!<�&��4=Mʼ�;������������<���<N4��M�<�����o*=.�Hȫ�K輚�[���;?t��XC�<z;-���]@̼fJ�=qE\�el ��M�nH7=���������������<)�I_��p��E�|�o�
��;eB'= �>��=TOռ��<��f<6����<�Ů=��<��=Ր���b����#���<>���#ñ��Ϫ��+�=z^<��� �<#)��{�&�;�=[VR�4��l�<����_�:a"��"ƻ{3�|�:�L=D�D;#�,�{��<��o���<qȿ��"�;/��C7��d0@<[�����<Q3�:�4=�s�Q�}���Ϋe=V�<%>��!;Xa��/�.<��=ۿڼ��m��嘼A����G<����gϾ�~=�<v��hh���=VÍ�󄱻�u�<W(�<�{<ȱ=H��<�T�-9;=ժo<52='6=�� =|w�<@?=b=��D����?�:Á<�h�Y?'<���`C߽��Ҽ�<h����y���泽C�S=jA���	P��j\<�P�F�N=@Ma����z�[��������=p*�<��<��;Hů��:Y=8�</����+]=fa�J&�<�(N=',�;,��<TX~=j �bh<�Ͻa���Қ�\��=k�����[U������G�)=�}9�4h¼9꺼e6���*�=��=y��<�*M��3ڽ��<�W�<��="D��7?���U������ƕ=C�=�핼��ɦ4�R����$9��?�=h#=����+�9��=�	�;��=�M��oY����z�<|��8���{F��)s׽KAJ=��=�?N<�f��6;���Ud<���<�%ƽ�μ��F��- =4����C�
���ma�u->=2
O<E�\�;'�x59���g�=�L<�w��;����.<���v�;��#��H���G<������?=A�<' :�mh)=�1�<ĝ��=P.��%C��$�<@M:IǇ��A1��R�<p�A;E� =5(����#=e&ü᣽�1�;�M�?��<��\=R  =��G���<wڈ��,����i;��f,��%���Tj=2�#=�E���;�;:��X���0=U�<�S;��d�P�;��@�������x����L�=Ą�1�<Iֹ<�F�ˤ�=q6�<�+:'!��6 � js��w2����>^�U�T=)���������j�^-L���=���;԰ݼ���5�j�ױG=1��M�D��y&��W<M�J=O��=����������I�>:�.q�C���o|R��� <��=��Ҽ�'<}Q��6<G0绶� ������g��\='� �ҽ1�<���<���H�|Ӈ:�*3=B�!=k�F;�8
�p��b-=~�X< Ԃ� ���$=z��=#������<�H���;�<.����4����;Bq �&�\���D�p���>4�� ��x��	�����<�L>�ᴲ�Ē�<,Â����<�4�E&߽f%��7�<-���|���˂<��%;4f�<�=.�n�=�j3<�=<*����5=��.=a㾼���U��;�9=y���"�:&`����侼�.A�<�D��&����=�v/��W7���>�(p�;�3�H|��**��������ܼ�$�<P�c�x��<�{m��	�<AG=�A�<�c;��<�י��=�o)=��=}�=�i.<�=I�Z<h����93!�<�
�:]x�����������<�����$�[���J��H=Dbw<�t������gZ<�/0�Q�7���5���䄽�F�:W%�5;����⼳S����`=�����������B���-O�<���il�e6)�C	��T9="�R;C#��j�������'<�:=SyZ�����k��M+�<�����N��@��䡽�0˻g�Y�}>��m<����g��<l��<L�T�*�6���ý���=j'5�O�O�ռ8A�<&k�<��޼"�۽!�"����<�q�9l<߄/�΋D=[������<L0޼��<;qeP�@���x6=��!���0���Q����@��	G�VF�����U��g�=T<���$�Y�T'��3==#C<������ü%v��i�	=��=:8��1�������x�G)b��/k�1�;�Cӽ0S*<����Zj��O�}�� �+=�ѱ�ZX�0�ּ�f_�]I����<2ٖ���̼B�����7�9��mM=�v�?�<|Ӽ~�<��¼��1<���<;k(=��
�`h<�p:��H�3�[��<�⸻ej������U;?�-=N�9�	��;#u'���Q�R���T���H2=�1c�R���D�;�%�<��y<?\��Aj�Ԁ9����紽[۶=<ȍ=�ƃ��� =��I���R�$��=r�;3E\������G(�d�C=�ָ�g�d�[�<wp��a;�[�&���x��v_�B����"�<_3�a��`սDZ3�3i\<M���}� ���J��=��A�������漠���/�=����=\�����<������A�"қ;n9:�=��jb�)�";�Q ��μ�,=�V�<�¼a>�'���nf뼕N<ir]�CJ�<L��<�gS�$e�:G.;�ͼ��=p-ֻV=6�{�r�"��<w<	�)='�$�#���=;k�;ډ8=�鑽����(=b������Xs;F`f��y���L9�����.��<C�����h��&�. �:�E=9���z���:��Լ/z�<Xpv�t��ƹ+=P�5=bB��1_�׸t�/gY����P�.�����v_�F�L�-�
=:�<yo?<*Ļ;u�\<�t{�μ���qz8�w_Z<��C<mm���o<*�?��q<�~��q�<�C���� ���;wټo�=o9���!���K��Z�7:R�<FF��c�=�d�qj7�9&�Y=*|�E��y�5;a��<�[��e=G���*F(���<�Y�<�'�����*#�<5C��E��η;[R����E=�XA�l��@c���^"�� a<��<���;�Q��������F���f��� �x⊽zz��u��<��޼���x����="�b�X�<ʐ�<CUi�?U=dż7��Ґ���~;� O=),>���`�I��<�}.��f�<�^e�s���8At��Y��g/ ��1�<����q��b��A�J�M��< ���c��ڻ�@�������,��<D_<B�x<�s��;A=Gv!�d������ූ<�$=�CȽL(=����N}�=�	=�F;[�>�@]��$�<���;u�7�� !����<����br;�S�,G� �*=�k�=�?�;1�%;n堻G����=?}�<��z<h㊽�����N<��W=4J�:�=}0*��0<����zֽd���ӻG��=D�L:/���"�T�q��<�+�N�;[?;����;�i�=�����J�<*!�l8���=Ī߼R����m�J�DW���7�:�%=�gd�?�=r��<sﴼ��|�b��<�P�ܘS�b��<��=�I=_2��%ƻ�E=�0�����<]�˽!j��]7J�MǦ���<�T!��ZE�x�;��U�z��s;��*7=V�<l�=�o����<,�Y=��X=�#�8�V����=?��:�X�J����4�;�Ѭ����V��=��G=s��}Й;��)=sJ;�$޻/���옮�ֻj��w&=]ln�%惽}�K=�"�CN�����;�4,�-a���?���,��,����>���y=~J�=5�`<�#�<� �k���Ȥ;�P�<E�C�����=�&�= ޣ;��R���/�{����G�&��l<�;O;������<`2=�a�<���;��	<�fI����<kjb;�ֻS$����2��y����������<�L<�̦;�T��ͬ<�Ȍ�/*4��=�����<��1=*Ҥ��=��������3=^��<)�<��A��x�{dc�ѹ��q=��8=Q���G����{����<q@j<�=G�9o�<��k����#�"<"��c����|�d=����^�(����#V�+�;u�<���<!����K��q�z�^u��F��8�.�F�*�T<f�g��Jl���g����� ޼ �D�9��^�<�̋��Z�=�_���<��!���A*#=�t�=��<Σ�<&�<����N�=�d�<�tz�e:~�g��<%$<���=w�h�0G����＀�_��T=o;:�T���n;*9���"�9����$��Cɼ(tk��,�<�ֽ�����<0\����>�
]B�����aG�5oɻ�Xv�&�����:O׮=�M&�sFQ��f���ڼ�﷼o�� �^�&x��BW����d���&º}��<s)��Qr=di)�8�̼�Aμ᧼I�;6�+=�a��lU��r���a%��g��;\�л�v�E'!�H�=J2-��ނ�kN?�f���ƣ)����� 쯼��<�=��{�Ҽ�᜼k��;Pv6�3O���=˷�=	����q��1<��ý�yϽ�Ӑ���ҽ�Z=�"}�$��Q���}=�R�=k��ղ�󖣽f����=:����]��*�֡�=	�(<��d�����񉽳Z�;�s��_Px<:�a=3������� ���7<�^��k���F��<�:=�����<,1=��ͼ�k=�/�A<K��u�;U}�<�B	=��D���H�rQ=����<H�+>��gJ
����6�%<�č�q�:����b==}]����2}=�4<�0�� G��*��<�e�<�/\��"��[o,=��2�<�=���R��Vf�;��z�_-K=x��=���<A����:������A��E�[�s\1<�/:K)��ٛ�:�<���o��<����=,p=CŨ���C����=�>ν��;�垽�ᇼ�r���� ��^�;$�����<���<�6Ľk�?�kQ�E+��:�<=T��Oam�X�<7�"=�凼�n��~���ɻ3�#=��j��  ;SZ��[�=�C�<��>�������:s�ټ�t��p�I�X=L!ɼ?c<u�<�1ü8f�;O���ը��Zʌ�i�=���߸��L���v�u=��<�3�O	�< �q��x�Eb�<�֦�1=cc<��	�$��;1o������X����C�Aм�PM<5i�<�5Լ:u�<�߼<�,�;$�O�X��=�m����@�����H���wӼm�ݼW��;Ady=�)_����=�><����D'����k�ir����=��Ͻ�&��Ƚ)��>3=�As�����+5���<�˟�"���{���9<�O=8�#��;�wH�<�ۻ1�=��?�l4�П#=]@�5��F �#��<��$��9�#Y��=�
��2��etE�95��'=~�p��с<�b��}Z<g�=£���S��6 =�忻�+�c�&=�RU;j9ʼ��� ���n<�<���.��;��hՆ<u�<��:=$Kܼh&<�]�<�*����_Hl���ۼ8�;�5�;W2��у�ů�;W����8���ⶽ�x��SK��,Z/�x�)'�E��;	w=����ߖ�;��6��ß���P=A�=��!=��s�1N�1Xu=�1�
#�<J|;������ۼzF�9:-	=�c���\���9)���ت�;���`����u�<;��<�T�|q�S��<���1\��vӌ�!Ϡ���=��<�+��_����Z����
=���
�<�1�;����"�<�;�Xs;��n:I�Ż)���m�,=z�۽h����5a�-��=�q�뽫�����*^��������L��٪���H��9����H=QƽI̡;�Ɖ=����<�<�� ��wؼ��H��<XS=���0��<�x4<{

��al�T����v��4���PL =�x�=����4&�شؼ�����.�����<�}>�k�K���3�ⷷ;2m�8=��q���!�i�<l�r<~�k�BZI�b�=�<ŤW<���E0���Ƽ�懼�q?��	'������m=�׼:~�<reɼ��=܂3����<�'o�fuT���ټ�!=B�;s��p��t��Sn:�ż��ѻ��	��@=��D�;�Yp=h�$���b���<��#;���=����)����%=��e��$P�]��w^���=���8��ɼ?�л���<H.��r���YE=��A=a�m;�{��Ū;ř�<�:�;l]9����,����S�B<�]�<w��h��<�����p<�����ȼ��x;V3O��6C����~}��F9�<�\�}'�梔�j̏<������B[=؞�=���<����@딼DW�<.�ļU��<���<�t1���<F/���5�u�D=���E�0�:�<�!$;uEּK�b��6��w=j� =�䃼��C����<���_�;<ټ�?�O=�Y,�	�r����<Du:����}����.<2"=p!½g��<ɡL=��<��k��y��⒉�=W����2��@<n%ݽo5&=�H�<-�}��<Լ�罋�+��*$��d=�sB���u�{"��<����WҼ�����}�&��;�]û��;�a�<���<��{g�ُ8���<�o�ɻ�)Ѽ��==?��=K�̽�� �6H�D蘽��l<�R�Ъn��q���ܢ��4=)6��W�U��<8���5��;�Y��<��h�T9=$�<��q��<H�'��LVB�دE���Լ�d������W9��'=̳�<��<��<n��;`�=;>�<��#���=��;���}���׼��K?v�ѽ��w �������=bV���V��鎺z�Y=�ڥ=7�|ά���]�A�:�Q4#���n=��$=o
���)��J�j��<�+#�Wʼ�����$��	���=��;b�üC���NO��@=��ڼfЬ�p,׼���;�%/�OR�u;n=��_���Z�K=肽��a<�F����&=�&��ث<��p}��x��Q>6���?=��W�+��;�K���$�j�3���E;Bf=|��=�����`O�Y5��L����=>�=ff��l� ��0=����)�h�}dZ����>��[񤽻'.<��B=8�ھ=w������u���<=R���N�~��2T��k�=M�y=�+�1�7����%wM<��G���t���;=�]a<�����.�>�e������)�9E��~�<mj�������<Ʃ6��?l<R�t��r��(j�<Bc�B_��i�c<�ފ:9�S�<"<�N����	ƍ=�`��b�����v �<s=P�d��4���м6���XYa<%m=����k�4��S=$� =��v��?V<�+�����&������s3=��j�İ�<Ӑ�<A۽v��pN�� �<5�}=��y���=,���+�:B	=�v�<���<�8�\Ճ���T<3�D�q��� �<�3�=
ڍ;�\�dp��������=���ľԼ���ƻ��<H)�<tk�<H������2={�S<Bq ���F��#F�4�(=	���Q:=A�;{I�n�c<���ռ��=�����`�/�=�QK=HƢ;�	�1}9<��<�5;�]�4%���2	�����AZ��� =��9C�1=����`�^i�<a��ؑ�<L����.���4O<�ѷ��1"=���˫.�p�,�8��h_;ܼ��<�"����<U�[c#<)2S=I�9	�=�'H�z3ϼ����"ؼ�Zȼ�iC=�=.�2&�����_{��%}�����������=��<p��=��k��Ei���}���|<�,��m����n��da=�k�ӆ�����c<2�O����<pW޼^D����(<(�6=��� �Œ<4K���ߘ�����fK=G?޼��߼�ݼ�*"��	�<^�4����<W�ܼ����{�<���<LJ;.�8=9��?6�k���9�<O�ƽ��<�9�:�����T���7=1�C�j<<6���(��>|<q�;6ȼ}F�$(�<�?żָO<��<'4�����j�=�SgK�[���s���<&A��G���m<�V&�Ȩ��ׄ(=4�]��qý�U �[�$�{��=�o,�s�<����>�l<�1C�B�l�]6#=H�M�2F <�l��������;�\�����>_��bS��(컼 =��zȺd^\=K\Ǽ�v��	n�T��,n��R��Q�޼i�$�ӯ={o_=ʽ�i�<gc�����=Z�˽Q��<=�wԽ�}��Ͱ�6�u�{�H���n!=�\�#���B�=Aq$<m 5�O@��U��i=*�VJ޽��)=U~�;��z<w��<-��I�;�4v=z�>���;ԓi�*ǋ�����נ�!����;Z� %�=�潼�@<R"���q���Z=K=h��=ю�=�ཬ|ýRi0=����w멻�
ܼ���;-2<;�M���'��z�<_�&=�߀�C�6������|�2<��E�ҽ�7�*P�=��5=��Ѽ�&�;�C<罧;�'W�m!�<�� =��ջ�����$��⼫7=�a�����;3��|�3� ��;�P�� ��&�<|!�O����;���].1�u�;X��pt�<Cr"<"���9[�=Lt����U='�x�(7�<��<��½<���EZż0F��H<�읽użQN=��o;�=�=�i<&T���E�=�X�����=�x	�I_	<Q!v�򗑼�K�<�e�N�F��p�`�~��<�_�;-Nּ�<K�ռ)L�<�z{��'���=Z�ݽ�<�4�<cdf��wŽx�)��둽�͜��L����=��7p�<~z�<�4���q	�^�2<��hˀ=�ݝ=�G����=y��,�<L�/��r:=���<�|�<��E��i�T"�w&Ǽr8���B<�|3=���<��p��)<F׼�7�^.��n�$1|��I?�[���[�^�=�1=f+=*[��Ѽ=�<W��jg�L=�%��b@�<6�=�Y�apܻ&����Լ��4:��?�R(�<�������<�EU=ݔF��w��@D���#&�>�=qƿ���}���.;04<=�R#=�)8���׻�bU=�e��8����p���;3x��T�����<����8�~�R��<��;A7;��<�3̼�8��j���I���T�����?>\=n�{�W�i:��ND���3���;����a�c|<[�<���;M-��R�<�No=߄=)y��-�lV�t�<ׁI�#�K�+� �o�G����R>"=�c��1w�+�O<�Qƻ�;�<~=8�V;�i��K���E;��;�һ�ص�<a-�:f�X�-�< ⼐]�<]基��: Z=��l��Sü���;X񌻿�޼��<�a�;����`<�+�}A��U;�䔺G��jMe<Yj��k�<�=��=X��� =�)�;c����<p���K�;\����4<
�t�=o��<C��<:E߽s�o�ƽ���<ۺ��!��髧���ͽ�N<E�;�%v=�I<&���U�=��˼xk���;pw=�-׻���𿢽yF�������_�PK=���Y?�Ǖ6��Q<W��<�}�=?�=`Y����7�̼�=�P���PȼR�=��R�� Խ�چ="��4�=�#���߼h4=��nTv=|ȍ;&?n��;�`l��R"��B1�O�0���G���;,f=�S%�r=��Ew���Ŀ�i�=*Ǿ�	N9�ֶ�<=xּ�y�;/&�Q��	^<�Խ/��<�Ee���:���<f;�đ=�;���i�<k��G�c<c~���]k��@���=�D�<Sc�0��;dc��|^ü�<�<	I
=V��<��>��;�r*�}po���%=� �=#��=* 2<�1W�4e�<��r<��Y�i�\ �=�ֱ��==6'=�����P`�n���=�[_<�y����;�iJ�6I�<�c =�o��A�;��Y�Ԉ2���=;X���?�(�E=l�<��<��ʼgG�<{�(E�<@�Ƽ7=����P^<�ϭ�}�1;l��蠙<�oy��-=K��<��;U�S=�W��H��ɑ�=�߽�B��� ��Z�0Z=����0ӈ��3�<t"�1]�=c��ދ��\�5�����"9-=�?=Ʌƽb��\��d-O��P;�"<[����[;��<\؀=� x���=
Xʻ	�<;���=��������<(9Ƽ��;`ͼȅ�8	������m)�� .=�9�<������=��;	AI���%�H�û�<c�A���=FB���;Gm;�5���6Q���¼D�B�-�w��<������0��F�=�=c����.�w)���Ľ,v`=\}�� Pu���<�9�t1��𷼯��<�,o�&m̻���;�9rA=���$��<�=��ý��G� �н�+���[�=p98=�x��t��<&�=�vs<�ؼ;p�����=�F�P~�;6y ��r�-���%L=�~=n40�p���������ܽ>��=M�N=�=��򼜖Ｉ5<��<��<\�= �z�fo�y���S�=�f�;m���(=>�μC&2�H�� ��a&�<�`�|<�p>!�޽��=�GM�����B�=�ݘ�4m=��V�<��y��=��<|$�=7�1���ν!ꖽ���sf�=��ۇ6��j�����<�n经<�c?�DE���<���=��-=ɷ<ܹ=��B)��#���6d���=�iռ0��YH:X��;���<h<8<��=��7=f\<�
��=��하��׼�bڼ�9�;/<�<ػ�Z�G�.�c=��<�gt=�����߿<B�3�^��N:==�<�/3�2�=� н��ܼ��=�7M<�>�<��T����<6��;L��<axf<�$��[�=�ov���i�3�v�	��PL=#��\*ҼI�X�ȼ�&c=��4=�ﲻ6��<\@�8V=t+ܼ�m=�����=�?��{C���3�� g��A���8Dټ���2���~=on�<���}��<J?:2rD�g7	�ౌ��;=��4�Y�̼>�ӽϲ<�`7z<84i=�ߐ;LW��kμ*�<F�<���zSy���!��2��pE<��0�K�=�<Bl�<voA��Ҽ@H��=����;��:4 �=��1:
[�;��7=V��OK�<d\νX�}����=��7=5��<�Ӽ��)��oѽ��=�m�l�=�T$����j���=̆5���u�	���=������=��)=9_�<�i"��=AdO=��Nnq�j2���<8��<���}'��QV��x>�h�]<�\M=Kݼ�a9������b���:,F���D�!w�<� \=�b��8X�<����T<��<ԥĽ6l��t���YX=xʜ;
RM;~Rh<U>���=<�1=��.���/��^0=��<��X�QݼY�
<�^=�a�<MH��fD<@}2�Ǵ���%��
=5U�;���6�<�Z�<X-ü`���{�:�3�<�;�4����?=q��<Pc�<�J�kҙ=˨<PKL=W�
��=ة�<��<f|���#��U=���Ž,�=��;������=�"=;z=P�_;I=㷃<�{ϼ ����=f�;�y���?C=�Ȫ��0�1�Y<:����<�1�<�^M��Y��?�<<��;j����<�+����<bTx<6����1,�;�;���:73��x��<��
��|�<I�ȼ؂�<84���Ĉ=/ߊ���ع��==f��S���>{�~X��<wO=ٌH=�H�O�<��޼W�*���#=*)=��¼l�?<1��<�D��&J=Ưp=�\�<��R<�2��A;;��'=�ҼB=��<�[ͽ6=U�/������=�@>+1�;4��k��_�e$H=6�c��Q<�z=3���̼T�U)��nɼn�N=��༱V�<�=-\�A��<k[ ���w���'���2<��~��n<n�g<��N�ÅX�]˼B뼢4W=v��5=���E��<�2���ߧ=�����g��?��*�3=�	V:V�=�r2=�)
��mǽe�5�`��<MW^=��;�:�x�T��P7��YD=,#��<��<-�<�U����ǽ��=b����A��-=�)e�tY%<\����%=��5=	5>~���9�c���,�T��=8�ཝD��z�sY�=TC<�>"}�<Y	�Y���
ۼ�u�<�/���<Õ<9�P=�R=~Զ�r�w<~
Ǽ�M1�iI�=��o���z�=���=oț=ϐ��R������L{�8��<�
=�;�<�Ix�Wޅ<��@��z4��CL=�m�;�;�� ��Yл�y��X�%�Xm$��w"�j5���3�w�ȼC�<>?&<�v�<`٢�t������S=_FԼ�"F����=��6�w��;�^�<�6<}�S�==�J^�l5ռ��/<�x�;�5k�2G�"[��&=�V>z�:���k�Y�/=}��A\ս�E+=v]��n�=gG=���<�����=�=����o�z�+��Q���;�3�=�j�= ����4��7��Z3�i���~D��5׼
(����<<^u=<�*=�㺼�4�<�> �[��<�]<v%�c&����<��=X�v=Sk ��;�<$D|���<`�=eѶ��鼲�O=�<�p�<�KC�����w�Y�����=���<�Xg��=`�I<W�ཇͪ��M(�J��C����]�9c�<%��<�=��;�=��W�~:?=?I=�~�<�$�<�2=ڙ��v3�%0����j=�J;���;��?�,-���:�<�px�,Ƭ;Vu�=�/C�5��9km<�\��&�
�Q��S����������=> =�ß=�C��;����Y�(=m��9��<�C<`qm�Sn���ϳ�P"(;��ػF|μ�0t=S��7�����B=���<�u<�� �~{=ں����=�kn<���:�n}��5=ڢ3=-ˣ<[P��g�<�P�� �<\�4�����v=޷ظ�x=EG;�F ��{�V�תK��B�hW$�) ���ԼJ4%��Г�c�<��*+����<��6��%����L�`r��	��=��=L��=���%�[��玽���i��PR4����3:V��;�6�=%�=��R=y�%��[Y=�� � ��<�]n�뗥������'@� ��*d�<�Te�"�<���<��<�|��&�c=:Ğ�3���\�b<D,��j�����6E=$s�Ֆr<N7>=�:Y�K���B<��0���/ʉ=�S���!<N�Ƽ�)r�Z:���I�<��<��<71�<��4=ٱ-���&=g!<r�V�����0��=�V=Z=�[�<y�:<�J�<����Ǐ���ve��٥=,/ڼ�1/<������r���=�E��i����Y=�X�� �_=S��<�΀��=��l�=q3���=��q<l<=_�ν��<@��;�{F=�T�}���Ť���;
��<�Z#=��H=|�|�詨���G=@BD�6ܽ�A=�tV�ܫn��=�<�l�=<�5;��V=E�H�A�9�=<3=��=i�]=�&=d�<��<{�=�W�<N��<�j�$ԼXq�;�<�����<�a%<#���v��<�삼P�=�g�<f$żn���<�G=��=!ߺp�=6��
=������ڻ���Źt��B�<�/C=����Җ�J�-=���<�c�;�(<?�:��Ml= ޼B�}<b���<|%�=S���@ͼ��ӂ�< 9��u���_<�Og��Ļ����q� ��*�]�7;�=��X}:=v�=|�=�g=��8�=kڒ=u"׽��<�
=��������A��\��;O�k�(�!<�	�;)<=���<QǛ����=�
�=׼���O�7�߼�胼c�����A=m��<�Žxn��_��x��<�:�;g��<	�*�W=��_������$=Ƣ���d��A¼�΢=�;%�D���<e�=�)���� =���<��D�9��Q=L�<��
>�l �_G���̀��x�<�[����=�݋=�1�f/���/<
p�����<��=b�g�}ż�3����=��M=�|�=a����<E�x<��;Ztv<F�:>��m�:oㇼ���}9��=2��<��Ԑ��iq���<�u��Ut;��D���>=��h���:R
/=��żV=��ͧ����\=pݼB��6���Z��=˷�I��<�9=�R1�>C8}����=u ���9n<�X漽�T=x>����(�~�<�ƿ=7=< �<FǻB�<���aѻl�=૮<_F����<������G�L�<7��<���_�<m��<*�6��\�ЍG=l�D<b����<��=�ڦ<�k�<3�<R��;mǻ�����<N8�>Y�DḼq��;�f=��A<�Ԉ�~(�<C@.;v��+�`<�E�<=��a�=��<k����>=N��;�ڇ=�������[j =H3���=�u�o�<��=�u��l=� 0��4=z�z��5μʄ����9Q�$=4ʼ�3=�=�<OF�<"�=G?�<f6�����<�;<�[p�b?��ֻ�q<�p���X=����qW�84��nb�=��<=��<)l�<�`='[�	/Q;���@�4<5�<��^�:���A�(��܋<]I�=�*3<���p�<�Vʼ��<V)�<���=!#�qMu�P1�<⾲<K�ź�/Y=gy"<�}޼#Y)��9���=,�D=��g<��<�]Q=�=����<՗<;"$=r�!�q|
�"M;=.����"�����<lw#=�WT<��9�.��;�bE�|5�=�U=�N%��̯<�D=x��d� ��$���0��d=��˼�<�J�;���<�+������t�I�m�c=� �=��=2����]�������m�� 4���ռ���<J�����])
���=*��;�"K��?/=�:=S�߼������9<�)W<�Gd=I���4_�;,��2��;��Q=[�����'���h=�Zͼ�P`=����:��v[���f:���<�b�<�E'=[�����=ƚ׻-��^�<V9�;��ػ�o�<�騼����p:�<lU�P����L<��	=���=�]�=H�==��'��Kh� 6
>��ؽ��?=II�<��O�*��� ���X�H��<<Ѻ<����i� 7=�}�<��λ�dx�� =[{$�<�'��ͺ<oN��5�m=R�>�=���_<����+��;��,<p+�<Oid<��zkA������8�{=�Ze������e�:v�H=~�`>	���H���I�݊�;/��<1��Q	<Bi��sh5=�WM=�ּ8hD�R��<�[<`Ү��5����/���M�nQP=ؐ8;�=R���<�p���,#=� O�t��<n��<�/�94/=kOc;����,H���t�\D@��; QH<:˽"$�<�����=�w1=�){=�w	=K�0=?��<Od�<�u�;L*��B��:���;b+�W�,<�-ļ������<ܼ�_ý0��<d�=�gq<G�v=�������b>�<�h���yԼ��ʼ >	=����\�m[�;��`��L�g���;�6�;/G�<��B��#=�����><���<����m�==�=T<=�~<�?->s��=[&A���<�x�H���}=ҕ�������-����`�=I���?'�=x=3��x
<�i�;��<��Ǽ��$����!l:j�<�
���޻E���]�[�@z4<B7h�cDp<�� �1��#�:=F'�CH�<Rei<�?�=������:�9Ҽ�"���o-=+����3=� =�n��a���I<�;�Q]��.op=�k8��|�=a��--�<�?ϼ�
,�[;���N�=��A>=�K�^P>q{���M)�+n���<(=&S�=%�~<]�,���v;�d?=��)=����W�b�Uw��He#�#r=������Q��Q�En��Z�=�	�_;�*lA�^�z<�4<��;Kb���1�<Qrϼsd����O�9�r��rf/=^�="��;V!ܼ�zb=�K����';�ք��Q�~ӳ�[&�<�L���'�;�C=�����!=��;,_�=M�=ϻ���;���;zƂ�^�<�^� ;"���
=���<�AL;�	�����D`m�Җ�< e-���2=A;<��J=8�B=��i�33�����`P������c=�|���?=�*�W3�<]rr=�*Y= V�9��-=��ԽՉ�b��<�L�=k���d ���{=8˽e1�����7<��C=�6�=��׽{$�=٩�<H];�~6��UH=C����W]; aݼ�����5�;�9�<�>=:��1�%;d�g;=���<��<)H�;��>��y)�?�:�,Ҽ�����as=��A=�Mݼ����G��1�<�P��
�V�VI�:.�E=��;�K9<���<E
����ɟ<� ��D4=`'�<KN�zy��J��;b�4P(��[��:�����a�,���<�'0<Z�ܼ[�*=R*"<��*=�=IV=%X�f�<~�0=c�
=�Ü<�5	���������Z\k< V�<
d�<�B����B�����@;�W���%=��J�O�e=�XX�����;^/M��� ='�˼��"���/��5�=�o=42��@!=H�p��1=5K(�V}�<���?�#�=��<��e;���<�s�˘�;R&�d�ƻ�!<�gZ����^׆=�bF=��J��A=��\=P�N��=�}=����� 3��fC=E�;����ې�w�u=�?��v����L����M�a\�:��"�^�1���;A�����%Å;��Լ璾<|[M�<��;���=# �N��v.��%���<� 8���<I8g��i�<=�F=�O��U��	y��rv=BY�<�.��=��B2>�<�=�zV<��9c��b)��!�<є�ܔc��!�;h^��a�<��*=�>��Xu�=��&����;�$�;��=��0;$s��]U=��=`���*�,= �I���3�l<䔼���y�=�A�<L9<��Ƽ�ڼ����ݶ���ڽ��ܻQ�-=�Ƚ�t����<����<yWl=|��=� @�M��=���<0^�)�!��O�=\x=�m3���E�v�<*B���<`'=0yټ)�bJ���-;s<����-�2�=o4�a��R�=?n����;�HY��6���2���N<R&��{c�=�]=>,ͽ?R���=Ҽ�=f�i== M�S�=��+=�;�|��n��1�,=�/>=7_p���<�f��l��?�=2�<VĽ��=e�=-2=]Oi<L��#�����<�޼&�=~,=< V<4Ѐ<��n=38<�`=������E�ܼ+��=H��;�e��#1=��/=X: �B��<�5�<�=t�=-��\���ď;�2=��<��<�)%<%�;ц���L <������q���U=2��<4A�Re3<�������:w]�9�;꟬<]e=.�5�VWN��$�<����@�\�D�=r�<ׇﻖ�`��	9�\�}�u+<��T��W=��=�;/=J�P�R�J�Rw<z�<`��<�n��ɼ���<?����x��
J��3C%�u�ʽ��=��=��=�^��䱼�#����l壼m<e�`��T�8��/ͺ��==���]?<�f�<t�=��<1F�<ʠ�=���=8�%�x>�<�*F<(�E��$'�P���z�<�nռV_�ԍ�<z���ꜽ��ż�f`=@X���<P�(RB=P�< �+<�h�����m��7ʼ��;�yT;tBk�ˑ�<$.�;Gߍ�{?�;�O=N�#�|��l��R+���ɧ<�6�=CŔ��/���'���8=&�S�Z9�ن8;\�����!<m�?=��<g5�<�s�=�	7<��ּ�s];S���uD��粁<�{�6e񻲘�<�J��|���X߻Ow�0������:�L�<x!=<?^g=�+���k#�>P�<�U���l�l�ս��w��F=A�q<Y�=[&��s�3�X�������w$ļEP���ٻwy�=�`A�I��;R�<�\��.׼�N�<�B�n<�e�<<%@;���=��d�<J'=a��D����r�<^p��~�<��=Q�<��=H�,=�r���q7����<*�:��`����G�!=��;]�-;����/�����zq�<>L9�"=�F�;7�d=���;�:<ln9�5h='��<�H=kSz�H��[� =�����B]�����=�yܼ)<�e��\n�DJ��nT�=��#==c���Ш� A�=���<Y�J�0}H=A�	�2<�<��8�{չ<
fN9�H=?��-�E���<�ߚ<4U�4�J����l&�<;�<��^=�B�;�;�ێ�퍖���=�/=.��IẰ1=]�=Z���=��L�U �����<4g?=WW=��<�	��_,���*;���<xK�;=xt�;h��<�"��ټ�g��(_�$��aW=A�<Y�j;�M��~=�i�<��<9�M�O=|� ����<����+��s��8�u;,�$=�P=Ӄ%;.
�&4���=):7�i�X;L��F=r��<mB=5��io=��='#=��"���� �\��O=.[��*ժ<�����<���:�G��r�<KǾ��;�<V�=�& ��u��9�<w	�=��*�#��<��;�P�<�*��*=��	="Oa=؍z���8<�|=*}����I�6��<��<�*�<Oü��<��<Գ�<:9��I����;��_"*�Wׇ=���;[�]�����tk`<�=M�Ľ˿�<]���"�6;����GE=r��?�<
�u���޼2�;憼pT����iw<���B8����<v�H<�+�n���h��<��;��B�P>�U�6$)<��=��=lL>��<�_������֊�*�x��W4=�?�<�������y�9����@ =H�=:�w�@A��Dм�.g=�����.���2=:�����=2S��|`6;�w�;7��;�k$�+s���b��K���xD��bμ���<Շ���%�p�=ԅ��c}=�~�=x�<s$���<�-�M�ؘj:�~M��<�L=����TJ���]����#/<M��=�D<Sf=K�7���=A�ȼ������<Lt��,��<@���-ټm.~<0x<���F=4�B��M�;��=i��<��W���D=�R4�D`���=F�Ze���;�s��=w%<���;�Xn��_=r��o���c�g<�fԼ�P�=��=>d}��BĽ�"�\�1�*�0=�I~=����+�$�w<���
O=5U@=E�=ӂмYP���K�c_�A*<�R��w=��%�����z�v����6��d�5<}�<����:�\���d�
�"��(����=��,�8a3=�*=��<�oD=Lrv��ʍ�=�>��K'���v
=�%��m=x���^�V;!,3�>[l�%�=�-�(a�*��;
��.��<���60#�FιXj��J=^���=��!�%�<~U��h�<�M�T{:���<^2��6L��=]����=�
<̞r<��d=��C=��1*e���=����6/6�em6�a���}W<�LμdU<C�d>�=C��X��<�K������!g= ��꒒:N�\�`�	=CZ�=�1�<z1�O�1J^�X��:IB��R;�뢻,�<�^=�<	:˾�<G3�K�Ļ��;S	�;�L%<�!1���N<9�@=�M�������<�˹�Kĸ�C=�<�F�����d�nD_=nJ���7<,!=0��;qI=MO$����M�X=\%e<D�=O3=5n�S�X��V��r�̽T����<�3��E;�=���g=9,�<���<���;�=~����<�����F����8<�R��K�(�e<S5=�]S=qm�<w��1�=���M=�[=߆%��wA<�&���:�����<��]=̵K�����sɪ�,g/=D�<<;�<:��;Р�oa��$�/�=@"��A�%���6<��<��H��Ɨ=��g<�d<��k����=�'�=D)(<��-���=/8?�b�<=,��<R����=��B��G��O�ͻ��[=�jU=�hd��!޼u���Տ<�J=g�<��oE����<�����<�<�=P�����<�6���I]�iǩ=���?�P9�p :L!⼙@����,;�,<���;���<�C���c�;�6,�NA��\ȇ<�7=���C܇��N黔��:p1�<(N��� �-_���gx�	����=�<j��<"9�<����Rܼ��<s�4E�;���<�x)=�lA��S`=E7<%�U<�=]�|��ս=��Z�H�n����}�<���c�;Op伟�R�!=�������)(<7�=��=x`ﻱ��a��<�h�JB�:6��+|z<
�A=D.�<�`���=��8<4_���ȼ()��yy��:��t��<d�żt�H=�]��H!�<���<Ƌ���D=����fw=6�%=���Ox��Z��_}��D�;��<��:�:= ϣ;��A=6ׯ���7]���o1�:@�1�n0�����=�<�;�s"<�=(6=ܰ�<���<�z��_����*<D$=�5��c(�FlX<1�<��s�A%#=�3=��f���J�<�#�<Cp
�}�=
�_=�jP�騍;F��<��̻4�.>��8\5�=��=�#�<�C���.=[2X�./��$�P'��v�<Ǽ�+�=�<�<�f=8!=Ƕ����"=��8����څݻk�<}s;�|��<��r=�N=Jڲ<dᢼK��<͞弇��<�?�V1';1����&�<��ټ>�|��S�<�Z�(����zT=(�����2���;�@!=c`>�3��ۇ�������<H.�=Z�=z����^�f�M3�;7=��*=�ۆ:��<��u� h�<䶕�"�ü=��4׻\\�<�ق��̓�YÙ=�1��;׼�y��xJ=�jt�v_T=�R<�:�rv���b=
F�=�=����@��-���N�1��O1����;�l��\�ɻ���;�_;om=G��<|�<h�=$A���!!�|5J>p2)�o_ջ��z<���<����<���[���E=��)=�Pܽ,a�<?#�:X+�=^��~�t��H���,=�86=v��<P���o=,<�<�a����=����ʻ���oG=�:��՝Z=;�-�F�ɹ##Ⱥ.!��I������<spm=���9��<�[�ؖ�=$��<���<$P��wRz����=vч="��=o��gf�<s'�=� ���;H�IM=6e����=VQ�.�S=����5�u=��O��д<�Wɽ��#�1�!���9���8�9I�Y�3�=q-3��k�=GG����'��*D=����>_<`@=��h<�@�<,�<�R�@鎽"IX�~z =�<λ|#?�!���h��b\���U���<���"_�<cz��4���
�0L<�DA��7=m;���C�<_E��t�s��pF;�B�<7ļ������)
��.��S@3=c\<�w,�%���n�<J�;�.�<���3v���i�l<��%�6��<߆T=����t=ʒս�*��&�r<՝���<�:"ع44e���7�f��<p¦�%l=MB��XYe=wl\�$�w; ռۙ�;ϻ<j`��y漃��=�L�L�N�Z譺�v�����;� !����<��/<�ļ/�<찛:	�<@�=�h
��8;��6�ZW�<�M=T��<��A�2~�;���,:��A:�<��<5P�ڣX=����<]�<�-컠�»�un��׼���٤_<�������qｼ|�=Ud���-=<>�����Pj�IQ�F5�<��G< �=	�<91</آ�"�!q5;��Լ8#� �˼��ֻJ��ف={�K�-��<̭���M�<bZ��&�����<�μ�<!=_"�=:�l��'�;Ps�<]�=�h�5�=U�P�t~,���<���6�5;x�Y=Ht���=��	=.�߼�l�;[�/=�;5O��<��=��G�rU=-ʻ #������v=
M����>35��1�;-��=��=����z���W�2q=��<cӖ��q�;�;�<Fv�d7�g;��}�<�����=7S�0��¢ʽ�ͨ=��`�s��=F��.�z=�!�<�d=�΂=�a:�,�8l��Q�q���<�&`=O�a;k�<�Ĺ����='�=j2�;�T���@=��u��&�oބ;Iל�l/�=T=O��Ζ��
9�4��=�����=N弼��"���3�X�D=3��Ұ�85y���u<��*�G_t=�s��D����^=�C=:A=���=�n�>�\�cm8<A��@'׽lG=��{�ɼ�uֻD��%	/�~�b������}<�c�~�����'=�?;���<�}�<�'�:8���\��<�l� 㹻�k=�K�_�<vG��PO���w������V����<��J=�L =�[��4!<�$�;[��(��<���<n׼��S�v,<��W��s�=��<y3�{Z�=�;ż^��<f�<�g<�5Ҽ�^��+�<`5���&�cf*��X#=q^�<!Q�S"��&��=\����۷<�@^��<�?���D�<j<�2��]�=�:�(��[�<���
�S=�k(�}��<�l;�R��~����`׽0z4=GJ��������!���=ͣ9>,��=�i=.�@�#\(�<'N=|�V= R����
�����c��;Ӫ}=<���c/=ɼ��<_Ag<ɢ�:��{����<�~���r����4<4����ڣ<�Eo=4^*��X�r�Y<Cɡ<�巼�+�.�K=(�=0B�=c��=��ɽC��<��k�i�Y�>ڕ��J�_�&'�D��<�Y9�VC��t�=�G��U�q&�Im|<�h*=k��<I�;8&�<��.�}G�<
1�>Լ��9<�؎<�^�<[�ڼ�	�0C=�@Q<��W<jɋ<�<Zj���?=�[����n�*��6Ψ�
g��N�?=ס��7�� =yS=�dU���<��;�{�=̯!<�S�<��I=2y��1&=�~���`����>+�;��_�Ⱦ<A��<*�D������1��=��z��!=!��~:=g��:��ἡ՟��y<���<���<4��;O�=��#�Ԡ�<��;�Ҹ<K�����ֽ�;��=�mԼ���Cx�:*=��'=-��;|A=",���#�=�,�Ws<� ����ۼЛ=�������<ʢ��[r=V��=k�<w��z�:'��%ex[<qN��F�*=�����=� ��n5�,��<�N����<ޭ=�`����(<�q�M��;U =Q�<p��<z�x<�7�������<'`v�|^>������BFi<��-=F�<V�<�[,G���=�Jv��.=81����E<E�=Љ�<�)���=�+�t�$ݼ�D$=���=q3W�8N<5�=Qp�]�)=�X���7<�����C-=�y:C��/�d�	�|H=��<N⧻_{�:��;Jm�<���<b����q�: M<�D��[	�;����x%�<]���6��n��m'<?=���	E��<��<X�\��<�:=��<�$���<��B��������<�[�<�⭽�y���Q�l��=��ռا-=��N<�;�7�<0�)� 4$:�<6'�<�=�R6��,�:ɀ�;4M��
��Y���Y�=~�ͺ��3>�ԽC1��|P;�;�<�ꔽT��;�C=0��m�=�T��3=�ȋ����<g�˼<'�O������<�J=��0<z�9'[��|O�<O�<8�L=�<���]�\�И=Ƽ�;f�A:x�}����<����]�Y�IH=�<�;хA:��i�b�u:<ۼ�Z�O/�E�=�`J��t<�Lk�C�;=L�ἥ��3yF=�����A.=�����=�j�=&�¼�o� �=W�=�ʵ=�G=�
{�����<�B鼯�ܧ�<�q]��
����j&�<�/;у�<^:��w��F��'.G��x��[ҫ<��<�9�D1*�=�<�/=���L�<���<��]�F�p����A���P���@��尽	
��Z�������)>t&b=o�G=W�>*�eY|=1WV<q�7����<h�ýH>�<eV_<��>�+6�<�v���a8m��a�<���<_6B<u�;��i����n�#��ͻ;��<}(=\���/һ��=����a��<`zq�>�����s;�%=TT�1�=�V<Sݹ�?�=�a��_3=���<6�;�#@�]O�s�v<���jT5��4���<=y�1��;�<���<��:m�\���L���=<�Gb����=��;3��:yG=���<!������ӭ@=��	��H�95q#�p1<o>$�:��<�0��C�꼲�K����<F3A<ya�黭��_�)I#���L<X=��z�޼-����d�����>^��<�>�\L�=VB�<�_��PN<��=�	�R�<����X��=퉏��߰<�9<)!'�����v�cF�<1�;A���YK;6�9�*y��'t�9#<A�d�F��;�(=޴�<�t[�P���<��F�<�#=!��<��<kQ�3	�<�gG8�g���`�=�鵼Uy�<w�=��S/=cm"=�B�=�����<�W=D#�F�6=��Ż�W�;$�.6�ƒ�=#E�UQ��V<G�>���=���<a�;=�]%����<�Ae<h`�<y��%��=-�w��;�K��@�<��j���M=Ū=�q���9=�G=��<e�z� �g6cKμWW��8-=�E=Y�P�?��;�9���(�<��<zx7<t`�<��,=��{=�=��&>?нT"��x�\\�<�-�����;)N�;j�1=̔Z�%���s�a=�2Y�º�<�+N��6=�4��)= NW9������W<`���4�'<��)=@��bٞ�Xѿ��O�=���'ﺌ0��7E<�к�=
�B�E�����=;.?�� 	<�<˿۽�J�<�h����<vF��相��\�؃;�+��=����=�B=��?�^f]���=�쒽N��;%����S;l(=ꪬ�j�<R��=���=��ʼ^6�t	��D< �6�,QP��q=�N��+#��g�ּ6ח;57'<v C��
���Q�-�<�o<bE<׼�5���}ͼ���<��<����=l�=��_=��7��-�>*��:M=��|<�Y8=S�;��D=�:�� VA��7=��1�k
�<s<�}OǼ����<�  =�Ñ��}V=�=I���LM/=K���VI�<�W��[ZB<�j>�^b���2=Yi�z�/=S� =V�G:f�W;���������d^�<u�;�����C-=i㼧������<FO�<�0D��@k����:gT�=��<���=9ͫ�a�=u#���h_=o�켜�$�F��<-��<�����b<ᙆ��滼��<�ޓ��)D��	�o�~�-�y��Ї�k�d=��=MC<�n�<����
��;���;֦��Fd�:l�<��F<���;"B�֖=��1=%N{��_<<��=�ջ�I�Q�5<��G������;kL=4xm�L����;.Ɲ��J�<<�?���Y<�^)�'�=i���2Y��J| =Adl<�m/=q�)<j0<ZlM<!,�����ώ;������5���A�8=��6�A��Z�;�)�<�y��?S{�TnP=���=j]>�R��BV-<�<�6=�f��P&�%�n9#��;�]����g�#=��e���~���z;��<Do����<Wn*=c�p:��(<�/<��d���#=ey��ve9v�6<!s��Mt�(��<��<5=�F�<UW�<*���W�=؎<���;�?ɻv%Ͻ^w=�݀��Ȇ�o=�rU=o'�=�gf��S�=�Y�=�����JL��Ý={=�
K=�1��$!=� E=�z2=��=�b�����&�<<8]�3���@�������Լ�o�
MV;-�=��,��t�=B�>�h�<��o<2�5���;����;�>�=�Y���=�.�<XVW=�в<SU2���>����t��;�K.:F�<'%=i�$=8��pެ���<�-��; ��q�<ri:֯3>L;��gI�= ���u�=��!{=��g��`��X��Wx�˕D�����b-��������F=ju(��<:��<�D����<����c�ż�Ĳ<0�� ���<B�L=�=���;�%�<�qR<�㷼�cO����<��B� �Լଇ��.%��T�����tU�;!C�<�"=�2.=�RF�?��yx��g��=	���9\=!%&����;p�)�E=�����=
=gX�=� 3��_��I����=S���ť	=��=�[H=��<a���hZ:�ڒ�,�<d3��E=bᘻ������"��Ϟ�����:���n�=&:�<r<�=�z���G=�y��)hj��h*<�֬�>��O#��#ݼ�">���1m��"�=��9;^f׺�H&=�);��=# 1�F@�<�����O=���:~���m���û߫R�X;@<���<�T��.�_�'=}� =��Zȍ�˼��� ���q���[<��{��V���:����:1��<�o�<n?<�=[B�Ǖ��v�;'O<�셼�=i��<���<��<� =�����[N�b�7<Q��=([�;kA�%-��f���A��5;".=Q�IEA=��_�K�]�b;�z"W��x=�#�X��&s�<7�d��ڊ<�=Ć4�d0Ƽ&� ;�!��_\=X&$�#�;�R�<��c=Z�N��*�<��$���5=<��-������<�=f��=@�>$b�_�<�-=(lI<u��2��<�=�<��2����<NGͼ�u�=�=ޅ�;�!�<�h���^=�s�<��|�3
=� C=���<2�;%nκ�a=#�˽�[�<�k�<���= 
5=�`�<�5�*@=��	��r`�Q7=�W9�{��t�;[!�����=O�=3Ѿ��Jl��� ��25;cCH����#V=��*=���<=��&꼽�<!2B��\��йC=J=l"�6�<Tj��}��=�RL�r%<����,����;�[#�~�&�j�k<A};5��=��"��b� �=j�1��[l< ��������;�;���=5<W��V^�~��<^��Fz�;����估�¼]튼5<IT���n%=���=�c=��+<@A=0�(�L����0���@�<\����氺u��=D�=Hɚ=e�h=@ =N'¼�`�;��C<,��;v�=��,�b=�����)�=�����.�1
x�w�A��D5�b~=%�M= �$=3L�<>OU=Q��90+�<FR=�h�;�J&=2⻩h��Z�<���:����;�=f�ռ�V2<g�A��t=�;<��W�:���=��D���!7��l�K\n�K��=�ϛ<������<�ˊ<V��<�!=�/��!=��x��A*<�x#=������D=G$<��=��`�M񐽝��<ZH�=S5j�Z	��¹�=P-�<b�8����G��4�?�S���e|�=qn�;%\=��Ƽ��.=p����0�����j��pl�<�]=�M<�=�t���P�5�<\��o��;��;N�>��zk=���@=���Fd��3<Z�������M,���Ƽ�����-���<%D6��S�;��T=9�;�S<�Š��	=��
<As>=�Q��ه+��i���B�=�����T���Y=oL�=��7=�m�=��=lҐ�:{��#��Z(�<[�q=`�y�ѢK�6ǲ�m�<��*��"��E�� W{��
c��Z
=���<��p=b�b=�飼�`<�:.�P���y��S����:1���^��]h�<d	�<+�2�13�=��?<�:�����6����ɝ=<S`���L=���Jپg��=w�<��4;�{�<Kq��6I�<f������</$��d��;H Y=�A���<�B+�Bu�<'[�<��k<_�g;�Q�<ΰ?<‡�K	�<�w:�/=<"�=PN��t��<�M�<r;	��3<g_��˼�,�<��S<��v���K�)������<c�<�!�<��<H��<*��<"������=�z<8�<�	�&X�<�߂�7�=\��:��<��ͻ�������g���=�\=�	:�9��
�<��/="�!�T'=�+=�?����<�F1<F���{-λ��5����;DF����f�.�D�b\��s����к��<�PJ:��t�b��<�oº�D6� �����:Y�+���<��ݼs�<�����s;�����[ѽ�o�=���/��<3@=)L=1�4=`��<F=1�f<�E��W�;�Fn�,��<` �<~���@O�kɍ� ��;J���3j���e=�_ =܍�H��<T#���G�</����;��J���ż-~�;�%��]=���<뎾;"���&%;�ͼ	jT=�V�t><�I=	�ټmL�,�	=E,��Vy���������-1�Z>v6�h>�a_=����d*�}�p;��;@�;<�9!�3s5��	��=� ���#7fB�<��<�(<5��<��Z=m�ؼgp
= �L�f@B�pJ���/=;g�<�.=����r�ޓ�;��p:��<��p��6=^�<�W��҆=���=� �ch�=|��u1 <\��b��<�M�<H"=�ۡ<�D�(t�=��g��և<�-˽	��;	=X+Z�Կ�-d"=��<��̮,<��<~B�<�M#�
9�<:*�L��;u��:��<�m*�2̉<	�&=P\R=p�<+]�<�*4:Jq�<|�}���)g�<�暽�/ ;ec0��}~=���P���<~�T;O0��k�<�y�=N疽�η�i\=���{���!���_���)�A�0=9n�;���=㐃�/3�<���=�y�q~�<�f��F���)<=b�;��F��A�����<�PV<U�ӻ����=ZS#=�h�;���!�ۼ}�!=�5�&(�!�<��-�B=)%�=,B=�2�<=3�v=m�<8N*<?h�ܝ[<����jx�rV�<XK��Y���������+=�R�oļK:9��;�m5<kn�:2�^<\&<e(���u�<4$��HS>��I��ۼ�z2�}��;�$̼D
E��=B��<r�K<NܼS77�[%8>����s�����<�+>��<���=rܲ=�k�ly<#:=�5;�Z�<�쬼Wݻb���'��h�����?=[E�<a=��?���T<���<�l<���r�= ==Ы�u�z<�h+������ּ��<�e�;t�I�e��<ŕ�;��<Z�=ş���=��)=��m�K��<�o��;��H=�>޼�\���j=$�=o��2��<�� =%A�;�qM:�O�k>���:=�=�= q��tn<S�μ�,�=�h��B=�l�Jk<�/���p�<��H<��PH�;��z<N����:�Y3=��<(_+=҅d;�ؼ�C>:�#����ͼWK��	8��
���J<�yy��#���������wy�=��&<+��u�I�p=�p;
��;�㼈���� �����-u=�ə��@�<3k^=C+üԐ,���<n3庀o-���I���=aw��2=꿍��/<<��<\�<0�;$H �z����f,����=F�Z��9w�� =�{E<_�ln�:����F��=#��ʻ��Y:���<�"������q���
���ż�;5>�� ���1h<v!>�����I�=�	\��X��M��[��%=���[G\<��m�V����0��!�<�f&�i>��+�	�=��+�ĭd�k|�;�������<����ZiK��o="�x=t�#���a����'o}�j�<��?�lj�󾉼(���W����g�⫲�dc�<����º�쨽�S�<�5)��u>.8ڽfkR<��۽���=Ya<�-��J<�!�����eP���=�{�$�</�/=�x�;t��<�C�#2=H���ۨ5=�＼�)�����jC�<�(�<~>��|�M=��ϻ1�L2<�n��嫬=֮��G?<<��(��g=��<��;�����*��=&E���L=K=c��<0���a���!y����=A��=�r?��p�:WQ�e:��!���*��;ݩ�=4��=�żf:ҽ�w��yK=��¼S����G=I"<\��#:��<8����=&ɰ�/4�;�]�=�Rl�:�����\P,���J���=`�ﻫ�|�H�x��ib���;��="w�;��3=�2=V�	<��D�n!�9��;�7	=B�<���<���g��R7��@���=��]�z5�U�<]��<O�x�����_Q��F���!�<Iu<>=�J�"����=s�8�:��5=�o,=rs�������H=�_�<t2=��ZK9�0���a�<�M�<�=<X��oI�"���$y<
�~����^���P��4�ֻ㒽<�S	�vl!=�6)����=]e<=٢�`!2���<�+��<?�.;���<T�н���J��4z��D;�z��"�A=I���q�57���?���ϵ�b��]�3��;!=�+�<r0�;�I���<�8�[��ph<r=k��Z@�{ґ������	�<;}�<��t=h�<�u�;Ra��pe^�^�h��$t�(�ּ/x���;���<dC�I����k�=��;=����dR;�;�<+��;84���
��;Y��c�<�L2=�`�;�|�:.��׈�</�<v�;�����0�<�!�<�\��`=+�.�욈<U�C
�s�c=�BT=Ln�<Tp�;5V ��d�!��;�Μ=�d?��a�;ܽ�<q+=����*V�<�=t��<�b����=�kV�|��(7�K�+�}��Wy�T��;J� �;�,=�t�<%n�:��<�α�� ���7�<��x='����1�=�!B<�ڧ��;<v�[�<Y(�d�ּ����1�=&䭼S�d�%Xm:�f�<��=��>-��<�_�������	�һ\��G\=��<�h�=�~�<��<�щ��*~=P��a?��
��l��s��VV�u������2s=��G;~#�� ��o�;iU�<��(�X^�<Z�>8=M���5K��F���'<U�#�S�F=ݗ�<��i=�,���1ռ������:�=�rB��{�;�`�=��l���	�֤�<v�<�ي=F�������S�\���8���)<ęK��B��ߪ<�EJ���p�X��<���=���<R�&�O�8�iC��U�4 ����<�Te���:��< �û<���Ż��;�V�W="����p[<�������=7�V��T ��Y,�|�̻&ۡ<�$%<#1 =>:�F�b��oj���=ә
;��]�X�(=��j;b����8��\�;#�ɼ͠��%��<�$��e��͍q=��A<i�=-h�=���<p�=�U1���E=WQ�ҵS��μ�-f=2~e<��<+��u���!��\���껼ܪE<%B=�b�z�����H=*r�̾�=^C'=#M��-=l�=��M<z�V<�1��N��=m��i*u�F�/%>�Y�;;n���է�7���=��;�\�=�п�Ƹ���x�u��<^�Y=�'=�<�ea��{�;�=�鼈�f�1Њ;ԝ��;��?�h:u�=
�<�/7=�ڻ;M�P����;/)=�<����*�7^{���Ҽ�>��4>"��:�[ =O�$��D����I=�7
<�5�҆��)=�`�:^�<����|��rם<DΞ��=j����F�<��$��ċ;}9�4�=���������撠�D5�<H�8=�衻޹�=&=�o�<7�м~y������1��B��6X�<�<�Ԯ;��
<lX:;�������ajF;�(�=�	]�2	�bv#�m��=��&�W�Y��[�=_R%<�|=^̍���<"��;��d=�F*�Ool�T�=�Y.�HM���7�~��<K
��w�;�՜�Eo���<4�;���۪�v⦺��;jA�<�.��!
�3蝼ӟ�<.1=�O��ޔȼ��ͻWA#<��8�A�<}7G=ܭ�=@���p=��<E)=��~����ޙ<�&���-(=��Ъ����;rIO�#�ǻ�����V�=��%<J��93'=��s��1=K�m<��y��<*_L� ��<{��h�E<_�0�ZP,:�=埦;w���z�<���9S=l��<ԃ���#��8�����c�����<}/�<����g�1=���������><� <r��<�1�q��<�u"=oջ8�*������&<n��;c�+=6��_P<~�a=�I>�	=/�.��#��J�p��~�<��+��
I�b)�2��<6xW=���;�S��2O==N����}�=70<˓G�P{���;$d=�5^<X����4�x�z����R�<��<2�檹;%*�<���9�KN:Ѹ�<>�5�wm�<��>�P��C=/�$=��?<��ݽXZa��N��B=�X�$t=P�<����������;+u�;\��;C@�<^n�(�ջY5����t'R<*�伖Ѷ<9����)���<��;��P=, �?|#<���q��=���'�Ƚ��&=p�3��ʱ<�<�R�=�ő=�����Ѽ�qc�HS;=|�����<Ӧ<̋��-�<�c�<zټ��=0!��A%�9��;�fϼ�B� ��<[JZ=*̡=SDt�|_�=wa���Ϳ=�h�O�y�G�����<��U<�I<0^�<H˂�H3��@=�5=W=�J�
D<=%*�Y��:�A� ��<��=��:3�����w�R���&=mx��7E��=����z=/��=L�F����bn�@=�-�=z�=5�=cڗ�,���v�R�K�p��;������ȹl�:� =M����4<m�2E=o�q3|�s63�j�(=��<����衤��q=ս��}wּ��?<��<O)<&8�m���=�,������w@S> >�=��=�>/�tE��o]9=�5=�=�<^A=ݝ/<[cҼ�(���F�<!NB����Ot�;0G"="��<��<�0L����I]m�]=�;B%弓�;�݉�r�=,��<�<�<wY�;/:V<"HM=�>�����<LT�=�ۻ��}:.�<O\����<����Y�;�%���j=/o7=�43=�	�����L�=j�]=f*ۼ0�Ϻ)�E���<s�=���=-�3=�{���<
��̝�=@S2<^oM�#�u��	
=�惻D?�r\�<L�<��6��z�<A;��C�:�_=ÿG=�����AG<�i�!�<e���Ām��֒�2.`��>4==��v�k������,<�)\=[�j��:Z��= �=2�e<��ƽF_��i#�x� ��0�=03���*���<ܾ�=O=�Z�<,! <횏;m&k����З� ϊ<���� �B����=��#=�輾]9��:��Ű<�Z�1촼�C7�US�U�<�@��g�F�(p�̤�<�US<�,=2��<Ċ)=0��D����8�=65ջo���X�ԽZ�(�n�7=i��`���t�l<.��=����0���;8�=�":��<y�<B�4�T��<x��<�U=Z�T������$<$��;�v�;�ؓ�{�<�� ���=A��=���=�����6�<������t=�Խ3�q�+>=��=EXI=��,=MA�<�廤E�f�|=���s=�x4����<L�O=� �=�^;����~Gv<��!�(.�<�Ɲ;�l�<F���M�;�A��,P�U���Ra��ܮ�=��[<D��<�	�<��ԝ���=��;�y<gg&=��=T�$<�]�<p)
�|�!=FN<�gS�t,i;@�a=�p[=l����=!�L�i;�Y�=R�W�Mf�9Eͼ�0@�U���x��<�8u;p͊�~�^=x��<|����=���<�ZK��>Ἥ/�<���<�D�i=aƽ˼�<iF���a����<S#�[�A>)b�;� *����<��o�ɕ�;�-��L*<�z�<r�w����<����B����[��	=�<)��=�$����\�6�#(m=��Z��>=3�u<>B��-j�=u�C=j؊��S/�Kje�e�ϼ�|��2�L��<I����<;#��<Ts@��챼�_r��E=k~`�����s'�;Z|�<E�Ҽ#�$�OX��vR���\���J�<���;��;��<Q<�<p�=;�;(Y4��?.=J*�m%����o;/�����ӽK��= �ٺ�V���N��m�=�m�=��E�W�����K�����F�[��0��A1�<ig2=�}�=r�!=$ͻ;�H=����\�3�����\�:��%=��a��+׼��.�W#�<.@�Ε�;���Θ��ns���l��6�;Yջ�����z�<��;$�㼋KL=}��<�O�<�)r�
gռbhѻ������=���<~�<ԧ������<� r=T�/<�f�搕�1	X=f���q�;򫩻|=���<Y�c� ���?=���j�<�
o�ä=�-�����>�;��<&&�<=�kV��Y<<�?�	m�<�=��Ig_=-=z=R=}�8\�:=p�4<8�N�U�F=5q�Vϒ=ά =醽~�wv���)����<|HX���:��8=��<4q߼K���/h=��E��f=��p��<Wy�<Ν<�;�ض�<l�a;��<`�t�� =��һ|��<lԼh������<�>�7m�<Xļl�A�������=U�v<�w �|=S1�>ɗ����>Q�=n>���"P=��;�7<��X=���zZ>�8��/콯�ƽ�Æ��0�>�T=�=ӽ�0:��=m��=�S�<��=��=�y^ ��B����K�71y� +�<Xc����hi�=��*=(u���ډ=pb <+�i�4�t�oC�<&�۽���#������d���յ�fM�>��r�F��Y�~<�P��޷����x}=H��;qh��6��<�)<�Kd���y<mi���W�;���=0�==����F.���s��:�;YS���
�<�f���μAb>���Ҽv�<4���j1�<=��P����;x��<�MK��t�<����\	� `�p)�Xp����W��
�;�#o=�,�8�=Vh1��T㻿�E<g#�<���N+i�W��<���z~�Nμfr�<�n�=k�)���[=H�&�2���vǺ3&�ͮU=�+���*4�Ĭ<�ۮ<NS���m,�f(-=f����O<=ɓ��� ��#���43=5َ�c� �c7м�D\=,�+=��D�׳���k��W�;n����<P����TU=.l=�� ��������|�<� Լ�B�=�1��kVC����=���6�=O�=��]��{V=ȿ	�T*�=�@���*j�=g�6��Ke<�Љ�s+��A�Q�sH���,����ob�:Ȑ컫�!=.2���ڣ�2�.=�7P<�`��yLy=�����`��\r�;濰=!��<��r�m�<�,p��o;���<���(�s�Y7(�U�sNn�Ֆ(= #=�>�����pV��eA���=�J2�;zA���;Uj���/M=����u��Q�<rl�;�0=�V*�l�張�<3
ǽ�`�yx#;Ue�F�p=}��A0==������A�
;_��<\�Լ@,$��p���J<칄=��V;�뼾`�<��wB߽�G;<O�u���G=� �ж�<l&K=���V�<k]����e����6��t>=�<�C%�A	>Ͻr�<�;6��@�<������<��iD=�Ym�h6=6q�<�F+<�!��^$���<S͸�V��:��"<Uܼ<��˼F�j<��:��=���I����<tIz���=�N\�?c=��= ��c�f;�y=��0=��=o&u��ߐ<��3�b=ZᏽQ�X=���^T��m��2�;Ys=�M����
��\=�;\1�<h��3�<��=�>x�*1<�扽��W=��<=X�����#=��	���-=%fͼ�-�|Ù��\ټ%�=߱=�.0<�8r=��,��eȼ"�ۤ�<��>@�9UY��kP��!8�z`�<�ru��u��h�(��r<���;�Re;H�W���7��{3=�x�=2@�<$�Y=(�E=Va���E��瑼���=�2<|�h�?�Ó <��`<vb��%�Y<�ܑ<��<��)=&=|F��/e:%9a������Y\�� =ŗ<av=N!���<��HF컖���ɍ��< =I)0<���+,����<M��7=����Y8C�P"�<�ԓ�91�<Tۧ;�FŻ��5<dt�<]4=����|+<�JF�<���17�^�<���2�G=/��<+5�<B/q��<��=����m���D�<�F5��T��(�E��zO<��q��G�<�B��F��5��Y��v̜�spY<b.�=�
,�j��>ۦ�wċ;1=.������<���⓸=3�<����\ͨ<����h�<͙N�Eۥ�!�c=���<zk��l�<��Z���6=vx=u�V<7JH=zTF�<�=�w�P�V49=g'���	<��=;�<=��=d۷��m<��_�������"=�+^�����䞽Ze ��&�`�D==�h=&�N�����I�<���T�7=Lh�<�&'��i��z?�<�%�=��l=`b �2�"=���<b5=��;wJ<�BS����;�ׂ<F�<��[<�*�<�u,��$Z�k(��o״�鼴<������Y#=
㼄�����P�%G��#<KW��n�<���<�B5�\�
<F��<u��<($�d]�����MPڽ�W��J�=�Ъ��`=��=�r��#2�;Y�<߶ؼ�J<�b�u;L�L�a#j<��$��P��근�������׻4�̼0Lo<�A�<4HҼ�H���=�<O��<׭�<�=qm�<���<���oz~<`=iy�<4�m=����21�; <�P��d}�A��(�<Y�=6�%��D�������m=�dڽ���<-���'�<|@��ܒK<�z=�U==���<�׊<^bE������<dEl��B;=iS:��v��6�����<~��:U�Ｉ�G���ɼ8Y��%�=9l=J�N=/.��e�<[�#�^n=I8=��L�R�;�`=Q)��a	=#3���=�3��`�;��ݼ�&�,�l<�5����#�V�h�����%��=:n��
�ս	�Ļ�ɶ�
H�=��b<�n���<h=
�ϻqq�^�p<��<������!��;=-B�<��<UE=�_��c�G-=6!�=KP=/1,�P��D7��f=�C��K�=G���䉻�ݑ�;�F׻�~7=��o��D<���<= =��S=��=t�%����=�x$<ܳJ=�4b�����ݹ�*�?r�gi���=%==���=!�= �<;Ϫ�H��ݓ�<��<��E{�i�=�'�7�:��D0�u�<�D�=���=�H�=u�B=�Ĵ�~���ʧ<oW�<�ѽ����Ư�#1�=�C��r-=w��<��r�ʟ��?�=,u���~%��˙<���r�J=t��-�=�ʼ����\P;�l�ԼȌ���X�<�:n�A;A=n�d;Ç��O��Ǵ����=�]�*��f4��� �໭H��؂��{Yϼ�*�<�u����4�[<���;����˼:ns�	�%=��\=�-���<�&G<���<c�=��<�I�<�'�<�Ҧ��K�<u��AW�=�}!�<���[���V8����=g�=�T$��ĽZ�?�;<(<��?��LI�`�8��=���<Ѽ2����ES��S9��Q+%��,I����<5�<�8g<Z�=\!=���n���v�<���<�Æ�Y����<Ml��U\]�E*������;�݅�n:��>=��ɼ��<���=/+h=����N�p6�<�1��̘����;��@<{��=O���[�<P8r�W��\������=c�����<�^0:G�0=Ӌ�<��<��=�0��%�K��6I�s9��<_<$�u��u��	Ԟ<>c>�n���j\;��<�`��Պ<�Rz�7D��=?5;�)����>���>�x�=S��� D����#�����=<a���<|� ���#;Y瘼� <`��;%S<Zl=@�#=E�<�|�<(e=��,���<���:_쎼2=<=չR���;9��Vt�غ;�T;�Z;߻��v<H��u��<�N�����w�<n<��\��=��z��%=��H��Z�����<���=TC�<a�=͓��!�<�="s��|�ü2�(=�W�=���<�ʌ��3&�@/���� =�5��\w;<0�;��%<��;��-�� =Rhؼ��=:1�<�F޼��4=�E�<FҼ��=�S=��q=��/=Թ��댅���*�E�ǼL��/ʶ�5�M<f�߽�b���S�r�T=T�Y=�n <�I�*��=�;�<O�n<K�;�Y���G������=z���[�{=ye���U�ۥ�=��h=Z�R�i��<+�9��<5e� �="1���U���Z����=Xe�<��<�&< �5�S�s�� ;~�<��ߺfk��p�=����|�;ad���}�=�1<E;=�;��:zh8�Lch:��=x9?=��׽�X9=�mȽ�䤽�2�:`I�X��0��<f��X��<ӇB��o=H��=:������$��<9���jw<=,=nї<r���vY����N�:hC����<�?����+��_6=�N~=��WV�=w��� �a�^�`6���V=����������<���<M�8���!���ϼȗI��,;���=K��!,j=�<}=��������ؼ�a�=�t:=U��<v�v��V���0=p�˽�6�<����@a;�����A���Ҽ�,@<��<�s3�U$=NS�<���bo�<�C<�u��Π���1<���ļ��X:��=���=��<SX�������q=����� c��<#�˼w=���&�~��t�<���<�载6�:<���<t��N_����弔�m�0��<�P�=lP��@�ҽ=��;��C��C���v�<� (<�8'=���=�nI<�p<�U<.`��
��?˽ ]���h>ރT=�\=�^�<Vh�]�<��$�B��<�+�</V���a�R��:!�<�>%8�CJ:��=ț<��$=�t=���4h�K���� �I�<�H��O��OV���#�e��~�;�s�=�C<��l8<aټ]-Z;�<�,����)<n�\<6�0�on�<y)�����B,=	<��Ґ�<ޒ�:XT=�?<E���*���[¼Z��<D�����w��X#�( =<�I�=\��k�ʽd���J�+=$���X��֚��(4��$�ʸ��y�l�!}=0B���g=-��<��<�����ӽ����C֑�V���-������K��=~�3��]�������<�m+<*Tg�~|p=�F�9�~Ӽ}�l<\yu��������]��=�1������ݞ<;�)<G/�<瞘<�u`�S:��jUI=.�S���<.E=�a ��r����W���<��;��o��l�<�����e��L#<�w<�Ѹ=E$�<�>_=r���#�� A¼��<��<�B����&'==��;�4����<Ɇռ�'�<2���Wh$�!�(=�_<S|)�W�<�삻+��<Ֆ�;��=f�<#��=��?=/��P��i!r���=�|�<�� �1�<)�S=���?�<�jr����;�o�ә'=�W�<w&�<=�-�;�m<_Z�;WJ~�e�v=ǅ�<��t=�<�� ��|-=���<}�}���н�&���Y�=��{��=&:�=��ý��T�V\�=N��������V_�2
}<x8s=șY�@�>���ȼ8h;���G��<	z5�V1�<G�I=v��=�Lh�UE����<|ب9�Y�=��<�U�=��lD��}�l=!�; �;p�~�&O�<��=�KN;�r������ʽ=��<v�8<^<��-"�������g���o��Y�+<�!S=��<\��=B^�h���8R�;
������l{�;켲�s=���<��a=��v={��h��<hI��+=>�S��O/��OӼ�m�J@<�y?=#��;t��<��V=Y	8��(�%<�}���}�<�k}<�b=]��;�M=�L�DE�b�<i����"�đh<��;$v��K�<bKX=�x�1�j��0=3�a=~������j*<l��9�+�<�_n��	=߇����߻�z_���=��=N1=���<8����<���������ټ��P<���<&4Q=��;Æ�<W�b=$%���<�W���;*=�-����Ƶ�;���=�O3;�1��=wf�ri|�p��;�1<<�,<0��F��]	P=�X�#�d�/����A����-9����=��	��=���<Œ=sҟ<�y���຃����Y�
k�; ����wW�6�p;r��ß�+�Kr�9|=q1����㻁^�=���O�X#���BD���!���=(󛽘%;p��񙳽񆧼�E�=dѠ<W8�<��N�Q#���@�;e{��XF���8�г޼���<���[-�2��<V�=y�m��N��W�K�t=����1I�vZ<�����px�ڂ<?�;���\{=t��.<r�#<�@�<�ɼg�7�@�Y�	�<���<p�<�F.�ݧ;�[�4�#<N�</ &=uo����ܶqa�{KC<P	y=��*�uB�X���aP�;ަ�;s1;����\S�<�e6=HtR=�=н�<+d�;x ҽ�"�n=E�_�e���j��<�	�8a=ez�=
!�<X2^;e⢼�d/�s�P�Ȫ����
���="$.<]�;R��K<�8%=44�M�Ѽ�k
��>Z���;�W�<J��<��:3�����$�����ټ#����:��i=��;%t�<�Q=�z���+<U�j��B=-��<�~=��9���7��\=eX�<�:=k\��eT'�%Af<�H@<��<V���+�r=ݨ<�켼ԟ�<g���C䵼����<�5���=�;X�n<[��;�^"=�I<ed��=���ǡ���)=n�<>�?�y�B�P79=0q9��˃�<ٵX�}������;�$�=��/��mn�QX���fJ<��J��P��ٟP<�h�<I�w��m=Y�q����b�5��+S�u�p�u�=��J�[��>��"��Jo=Y''���ڽ�ր��Z��ݯ���Ղ�琦<ڜ=fv=*8C��zj<�*=H�+�.Kf<<U�5�<�,��R;)�<�s�=�� ���:t�<6�b���-�BLT< �R=���<�\"=31���=ɺ'=��=ۃ	��䅽lU!;" �њ�<9�=��mr]=�=���;��X�)ϼEFĽ�h��)O�=)�ټ>��=��<3x�<Z�;������H���K_=-ڼ*q�<��=�L_�-2��˼:.*� b
=�����SF�\=���8z�Ҽ�H<N �נY=*����.=��ܽ+J���(_��m�� �=��0�NJ=�I��<K��<�,�<f'm�&��<c�a=>+C=�S�<n�ñ>���8<���=t"�<]������=@���)�=��M�%I��� <y��<�m�|N�:��<n�<��M��4B='���9�v=9!�=�$)<����$<Uq
<��ǻ�V=��������-�@ǳ�C9Q<��=��;C�V1�=��=���J⥼.����<]A?<�匽�F<p\�DR5=➿�U���<�Ǆ;��;𹪼_�s;eot�C�g<�<|jX����75H�Mޓ�l���3=B�d��M=֫��M�Y����M<�â�&��3�3�ݾ�]��N��O>��:�g�����=bJ�<�VZ:��<9 >;yf)= /�yʻ|�6M�`}d<��D<#p�=N���|���X =b^�	�I��<O�<�e��)�;�Y�e[=/[�]�x<������:��<b_K�ʏ�<Z�P�f<��_�}��;��m�PW��6=�dO=nn��
�<�(�<���<�=<�G��#F��2�<j-=��'=48h<�9��"�t;�n�<����b��;SB?�c�C=��e���=Q�<f�ɼ U=nz޼�ϲ<� =��j������;��g���<��-]<v��E6<�����L=�]��ь�:�><<�J=/Hμ�:�=3�ͼ��f���-��rb;x��;���<"�;�c�< +�=^(2<�j��ja�;�9����=4�=�S�<��x=��<s�M���Ӽ��,<O�<W�6��<��Y��=,9�;H8�>ɹy��<n��-�H;�@;������@��=�m�jrG�=�0h��L��ֻaR�_��<���=�j0=�,�8YT�<J �<��
�l��=�	�=�DK���ǽ2�^�Qi<G�M�P����w"=b�Qp/��u�<�,8�½�=���=r(�����{>arM<��K:�+�=yOh��ˍ�	x>='Ί�Jf=�Bл�%ʼ&���[��_>��=15J���<�u��B=��B���\��H=ȩ=;���:�=:�F��g�:���<�J'=�;��d=��^;���H4$=bx���ˑ<�d��W�:t�=O'��#�>�ƣ���(=�/���<#<#1A��t�<	��<qv=S�=B�<�ݬ�O����3���� =�Ѻ������= �;�p$=�ᏻT�f�$�9;v��sq�<�F���=��<�>���dռ�'��^����<=
Pr��!�<����Eο���1=�7!=�b��<����.ɓ��=ƨA���=�ؚ�c���j���=<��<��;�ej���8�6�����d�#>3�	��Y��Q[=0�Ƚ��(�� �H�;
|��d�U�����<e<�<�X�9��*;�7��߀=A�ۻç�����̇������Q��$Ǐ<g��<�ڜ�2�=W�K=�����l=�e���X2�U�M<F�F�^!����=JtQ���%��Ļr+ڽ�T�����<��=�<��<��E<-kE<��ػ�F�<�W;8�'=Q����<(�=�����|-=׬[</�5���<���<�]/�_C*��ϰ�	z�QLg���6< �����<)a�<�$�=��p=|>���W�=u��<������0�%�����q�z����:�1=r�*<.��<F���z�<'��<��<mh�<Օ��N��}7ټ�v���K=]�\���i=�l�����W��O��!=='fj������<M��xB=�*���I=F���w�1�����9=f A����ۋ��>H=��/�-�|<o��<�v��8���������:�J=�B6<��?<����l��=4����/)�����X�>;J�����o ��f=j�="�==޼ ���Լ[�:i��]<Z-�<jԫ�R�%=�%�<w�=K=�;ľ�"�<h~��WOe=y=�"=�Y�=�h��"�y'��hW��ت=Ah��低�,=�e <2���K��<�np�1lT�J.!<�}8=��������<�<;,=��f�����<�&��*Z<mW�;���'��;n�x=P�	=R].=�^��z9��.���6߼���<���=}���(=0|�d�=V���<<��=ѷ ���:.'Q�@���h/�t������_�A-2>��<����C�|=1����
�
s:�Rk��H�<�a�;��X=���� +ϼ,h�����;�1=s��n^D=���� �=�=6�w�ټ��ػk�=ŏ߼J#ļ.�<?����� �K���G>��l�R����2=��f���;=񦇽^��� +���M��xܼ��=���<��<��=���=%2���+1=�e���˻A��Gs��F�Y��=EѼ�Ug=��9=@�Z��\3�Ԥ��%�<�����<	�9=�r�ࡾ�h��;
=��=0S<�,1���;+
�������<�d}���u=A��<@7D=��=��n<����ȓ;��ʼ��5=T{��ļ�䯽�o<Q��u��;�ɼl��;�y�<�=u��<J(g��ּ(D��rv2<���&?=���������<k�q=�`�:K�����;�M�<-�<F�@<�㜼2�Z<�\���˻�A<!���N�OH*���R���k���g<�h��Qf=�b�=�����'�ϊ����l��<���<K_%<֮�=�Ҽ[����+�=�2.��|<�(�ˁ�=��q��=P0�=��K���<���M�ս�I�f9<�>�����<���<nD@= q�<3X�<���<8$=xր=���hb�:���<������==O��=���<#�|��[�<;Y�;��h=��D=�R�Ǐ�*�����ܼ˰l��x�=�l>=t�1=�i��70=��i���q=?�ݽ�v����;�z�����_ֽ:ƃ=��0=��y=�-�=e��< �;�䭼�����ؼ[����Ua��k�=5㮼���;=�� ;��<vV =հ�=u���>�%y�<����97!<N��<"�;����X���(���t�<둒=��;�=l	�=�(9�A�Z=�+F<���i�6�)G��O��=X}���N<>���{��T
��{�����w�;��}=���2I<��߼#_�n֚<�/<�'��R;��;��z<Cv�<�:�Ƙ
�'�<j��{y�޶����缮���AV$<�9����<�#�=GT�<�a	=�s\��W<�,��'U =�����=D���	=���<�zG=�~��0�a���\�=��i<��V�*8=���:���NJ@�ל��􁅽�h�l)��h=��a<�*�=�np=:ϼ�<E�߻�/>���Ž��	;�e�=$K��=���<��_<t�=��Z��{����PCݻ����Z�=E D={���|=��켋�ռ}T=�z;=�ļ�
�A<�Ͻ��=�\���s�!�=���=<��=�����ͽԋ���<��¼g$�����_J̼o���-7��1z;�]5�,G=�>;0$�<Eˋ;��=<�^C��L=����Eҿ=Ѽ�������D�V��
P��7�<�Þ���优Hʽ|�#��8�L֫=4?=���<:��<�=0v�;6[�㕄;Pш��C[;-�<�����؈=��l=p��9���RG���Ľ��a�]�n�Y;���=��<4C;�	<�J7< i�:"���ڼ�+$�`���HZ������-�9��<0�h��~�ze=�璼LJ�����`0=t�y=�����<+ ��l������@	ʽ[`=�!�<n�?>�a<���;��<y�"��������ջ.�� =�N����?=���P�������=}�<<'_=)�l��p����޽- 0���=�ҍ���\;���S:l�eZ =KJe;��	=n���)գ9��X��޹bR=������M��墽jb�=�6��z:�<X�`=���Zt<<��>��V��wĽAT����绣�����<�l�����.=?�����[�強�����=�cҼ���~=;u<�6|���<r������-=�Pc�E��۱�<��d�G��1>=��<���;�ҧ;��k�ÝN<�;\���g���=c��<D^��)$��蠻��< 
-�if8�Y�A=���=W����6�P�/<3���W�˻%��t�<^Yu=��ȼዴ���0���J=��h�;���d��;i]����U=b��{�����Y�J�<�w�
h�<� =�/	;�L��8<��Q� A+��)�<S�<=-�q<!3<O�%=��<^����=���;S=@<�8�<�H�<�-M�`�=l�%�p�<���$��=~�:��> =z�i=!c1�>���R8�LQK=�M����y<	.����j�μ�jV�}�I���<���d�H��+(=��=X���������b����<@����X����'
L<���<�$��|=�&,�l�<��Ž
}�<#�7�&�Q��<�Sd���ϻ5����ɕ��=S�O�_�;у�<~1�<.:t�����ц�����<�ш=T�`<��=���;nH�{`�W&�<��z��c=z��<��<)<�<{�ܼL�d��-m��	�<����[u�����}=%�<H���|�]|�;xK�7�T�=Ƚ�;b�e��� �`��*,]����<�=>��)�t�ǼvL��,�=XH/;��<R���/�=�"�9
>�ٻ
2?��=�;��½�AR��M¼�J0����?<bE>�<%���;ۈ���v{;�Լ�Qw=� ̼YI�;L:�<U�;����T=��=Y��=Ztʽ�� =������<Z�x����C��W<� �F���;B=�=(lټ�<6=G=ԟ��J��5_=@Z!��f_<I�C�?�2<�N�:�Au;�fM<�?;�m��8��=�9��܆c=ԇ�<|�`�7� �����J���e�;`	a;#\=,h<F5�<���<P9���&T<82���=��<�z;� a�����b�[��8�9=�V=M�B��OK�<���; -�M5�;�L�� �Z=���O<=V�a���n�H�a�:�#=&rk=���=G��<�"?�7 �S�.�����={�D��=�O\�<�>Ry���(���ͼG\=�'2��H�ʅ��;��_�n<,/�r��>�<Z-޻���Ff<Ϡw<W�N=9B��+�c=h6�߃B=:�;�sѼ��G:�H���=�_=7�-���<�<�=�h<ըG���<�iۼ�s<mI�==_���1���;8g����<۾Y<��<�����4D���=��=�.{���"=�\e<p�=���<�9F<�=uA<<Q;�W<&?=�[+=�8���W��뜶<��I�q����6=�#�<ZZ<�N��`���=���<D�!�)[�=L�->��ھ͕O���;�����<��:��!I<^xV�?T��`K��)�=~�ۼ��<4:{�W!g<$�ͻR�u��b;���-=9�5���:bP<�:@=id����<<oƼ_�;���<N�<�vm=�$�<�ʸ<������꨿;����ȼn��e�:=V<-�����4�t��������&=C�<��u=z���$&�<��P���ޓ�ű�<E:�~c�<8	V�Xv�<��H<���;]=ɲ��_ӹ�;	����=4�><k�=��<�f�<n�=�%=Vg�VJ���!�2���p<��_���S<�0�>��7�6�P<�=�#-=M�R<0��<�-�Wռ�|ּ觻��%�mh+;��=�皽�񢽧{<`�/��}�=@�<J�<�;�S���<��ѼI�n�(�6;��3�ӟ9=:P<i��<n[�;h�9<`n@<���,���3H,��=��(=i肼&`�;��������(#<�j��f�D=��ӽɠ=�0F<X�c=��e��)���{d=]��E#= Hm���=L3j���Ƚ�����T�����,�=q�=#�v�Ϊp<�������<19��/�<��>3���ؼ٭F<�n=ud�,o�=�~���OE�UB����=<=��6�ߚ�Dr����^��<�=��s��<�ӽTe=z>�o�=�
=����^Ѽ"���}CY<�<��5�H��;�T<Ƃ4����u�>>�ͽsހ=X��x˼)�ݽ.����El=2�A�=�f���?=og�g4=��f;+5K=���D=��@]��Iƻ�3��0���O�Aμ�8(<��<5	�)0���̼�6=<���o!=��h���@=��n<~=��=k���=.I)��<R�3��<��A�B��<���<�;��9=�i���<3�=����SR$<��L==��<k�.<�?P��ݽ��,�Nl�<rl��&����2��%�=�XG�R�=�M
=v�a=U��9�o�8�S߮={�ټe��<ɳ�9����,����=�#W��!<�'Q=~U�<�?����<�	7;�e�<?�ƽ@����Ij<�/6<z�=�����ܶ�F	>��i�Kn/=jZY��/�:�i�N��=Y�=<v�=�� �aOb��냽��<P�ڼ`][;��ü,d<�f�;.7Іr��=�<v�<Kms��+<�>�=N��<�g�<� ,�1�"�X�;���<W\*�!�
=7���F�<����O���t�o)V��]~��xM��T";~�Q��8<de �Ą<��M=��<˄�<��м%{/�àt�8L�<�|��:e��<e��=ަ�;�	=7ල9��.�ƽw�\�\���R<��/�<��<���=�, �o>�ے=WV���D�<���;:��;���ةC��K�<Ѝ7<;�y�����#��g���Ϝ�:5=$:����5};=`>(=S���r=[Q���F�;�ʼ׫���'�M�(��!�=NW*<����0�%��p��_ʈ<�3)=�Z<�>�<�A�Ȝ�<����һ>�,<_�"=X�Ƽ�=<�o��v���s�;�i<�ʚ���<z츽�.���� �;����;u]=��<=��>?�[�>#H��D�=e�4�A3�<P��<�Ѽ'M�϶��x4;��9=��A�gy�<lՖ�{�֛�=��+����Q�޻�+z�����ߙ<̿N�wԒ��f�<�*<= ?=�v�<Bz:=4��<�l���޼C���%��VC9
��<;=>o�
�ၽ ��X�üg �ֳ=Gѵ�R��=�N�ɶ�;�I�2��"?���J==X�=�n�=mb<�ٻ=Y�ż�{ýz<��p�`8���]B@���F=����?(��n��n��� L�=�Y����<?=	Y���.�<�"�����߈�=�*!=3��;4m���e�oa��h����| <�4�=�IM��N��,�=����S�=	�:�(6<�b��I�F;��p�5�=��]=AiV;N<I�<(A<n�=��뼎�(���;�1= �`�d=��:<�)�<>��<Z��#�U�k6|<3��<.ᑼ��r��k,=?ea����<�ĉ;%}=ǒ� ;&<�X�������<R����y=���<�6����;�՞;����Pa��˽�Ҫ�F7=��<���=H�<Nw;����j�����d����5H�<Y�='`�<�}���=f�;f���k�<�݆<Я�=y�:��b�����<^��7��\>1��z��r�<p�\���E=�쥼ל�<�Ym<� �����<�h���$=�Ek�;��=<�i�<���<t�c�t�L����W9<�=O_��&����Ƽ����=��;<�vP=�7=��
>~V-=�Q����8�Z3������ڽ�ȼ��P���>,�/=yn�����<����]=�=L�6�;x�ļ���|j�:���:E�gǻ�.���B�;�¼Ab�<��<��P�^��e%=�8y<㠪��!=^�t���<�ϩ<���<�_&�uf�j ]���+>�r����<��<��X=�hƽ�,���|�ﺘ����]��6���+'=��<Ne��I���bH=�V =�۠��Q����i��݅�4�2=�!����,�{���:`=b9ݼ0E������Xs� �9���<�h���;��<��C�w��<��f�-~/�$V��ˍ�<	�\;餼{ʀ�hXR���4����< �=��ѽ���@�򃗽[C�<�H/=�F =Kk�=���<-FF��T�j�A�Y��=��<�M<�}N�ѧ�R�X��a#���;�e=Ӳ��=��0=^�g<�&�<�?<=�~���G=��m��F��7�����=<G<����ga�WL�<�=ySν?�m<T�.=�!�=	�vܧ=������
e<���<��=S�s<��Ҽ:���D��6a�r�=��<b�=gj�;��;c�*���2��7���\=��*=~t�=�K�Q�	=��#=���<�����H�YG�:��}=���=xI2=�3Y�+G={K�;���<��Z<:���O�X�1<<�M�I�廏{�=���<�⼽9�=���$)J<ܧY=��=;=��r��+0�#t���י���4��� ���B>$����=F쟼�=�D����7B��+�{Y�=%���Q�(�R�.�NI�<��� ��U����݇=%A��b�<�һ+�<|P�;�A=����iV�yf�<���<�s漵n�=��i��m��%��Ze�<����!-x=z�㻱�+=ӋM=�Z�=toּ��O���#�<V�(=��<Ị��,;�ɻJ�����鼨D������"���<�]���ʍ�u�p=�8û�(�`,��\0=��Ѽ�%��~=Ί�-� o<���=gh0=b�6��"�?s����]���<	�^=p��;\yj�Զ<׀����7<9︽�z�=�U�<E�==|�X=��<F������:;T�*=�T�<Lv�8G�W�_ͺJ�<'=�6=<^𽑺f�]~�=Ե=@���CJ�=%>����;�8�lg=�a�Ƚ��T<!���<B'=͏�<���+�g�-��<�<<�B<=�@��({;�UA�X
_=��[=A�=�-３��<���=s�7�p��:�������#�=���=�ˢ��6�
zi�}�����;� =%#��:>N�I�<�F�<����T����R=F�O<Ѻ'�ې<x�t<�}�<��l<.B;��мn=�)�ռ�sQ�PG���P=F@��P�<����"^<C��<�X���k:�����X(���ս�i��`�u=��Y>��ܽQ���7��=����<��V�%�<�(	=�&+<���\��Ք���@=������tR=l)�<<�5=�򐼇qW�_�y=>�U;bD	��f���<:=Fo�;YA¼iT��X1��!�� �;�O<�7�4�(�n�Fj=�f�<��=\ls���Q�&�ʽw{9�=�"=���<NS��Q�<�{#;�P�<�hE�
�.=��=`OU=!$ѽ/kȽ^;z��%ټ��;�^=+ �<�c��^���KVl=�T�<�4�;���<�����3��85�9ɼ��̼�8�:ɷ�<(%8�{)ϼ<%�<�pU=O��<@��f�˼�L�'�};=�U0��\<Ii����<T�c�g<��ݻm��<����3$�|p�=w#:�9��gj��ղ=�S<�ۉ=n)�is��c�ǻ�8��)�:�2�9��><�c=̦�=�Z)=�����n;��5<���9#�����<�����<�l�f;�;R��9�"T����<�Ų;斨�]�ƽ-@T���<�܀=�T�=�n���⼣�z�rΆ:�/��?'|����=���<:�۽��C>x�8�hA̽��[��N��5=�P]����<���ҫ����=?�=2����z��H	>{�=���=�zX�!�_�����4ڻ���~�*=i�)��D^����%pĽWld=�i�<�U�:H�=�{� �=P��d���ŉ����W�G����ކ<����z�<�?�<�+3���ﻃB=;¦m�L��<;Q���(�������(�L��.۷=  ��&�?;��_�7���꛼�@<<�+�<����7�<��ļ�V߼@ӣ<"��<2鼾'����]�� V
��(=��<[/4��.�:�~(�JD[�{��=ػ=R��<�v-<�	=,=���yz��؁=_U�����<��=��7��};+s;�U^�)��@�s�Pp:<w�(=�aܽ!�=gD�_�L=r�	��%�Ư�#��_�g=��x=�}�D��XII���@���1=8Y�=�z���g�C�:�ܼ�Z���
=���<���e<��|i=(�W����<A$輴R��F�<� ,�o�=Hۼ��;�JԼ������Y=�R�=�s�I�O<���n�%9��=��Rs=�Ƽ�lA��C=B{'<>��<�x��ъ<��=zr��n���4�jo]=:��<	gZ;�UM�,ɑ��;/=����Iպr��CM=U#=c?:�����go<1~��3=V�<����8�<�2=4�=�ͼG�,�>��]��S����< ����3=����J��tڀ����=�S�=��i�9[�����Un�<�-u��!�<���<��=־��eQ
<[�j�7��
���\R<⿟�� �s����_�<���<ف����C�N?���v�5]F=�F���;ự^�
�P=U��p����,n�b =��(=�K���<v�<��=3��<��n:�|�=�,ּ�q�<l��<*5 �E����a^]=bP�=J۽)�;�,#��I��?=o����ۉ��=���5�!���9v��U=�����<�̿;�@g=?�5=Pz3����<L==��;��λ;鐼W�ȼ?�0=�:�z������<mT'=�x=�%�;��=W�_=�s�<�T�D\���3����A=����x���3�d�~<��q��=����@��������=n�v#��9��Y�=�R3=DCH:�:��y���@��_��ns)�Afk=_uA=N��Z�<<=��.�<;iP�֒�;�kL��)=��=�c�z�<�����v=S�Z<�c����;��������bL����U��t���t����	S�R���o�5>q ��S�<�{�<ſ	�\��;��܂�<�����v?=|�8�����9��;9q=÷<=�Ι<�����D��ы����=�λ�@Z=�=i�ڼ��-��W����qʑ��_;7�=I�ԼUʳ��A�<]���x���b��ҫ�~P��-��<#��;dz�ɠU=��߱=�=z�=�(��5��k�V�h��=d�=^�����������ơټ�oU�i#<50<�Z���i�]�;;�N�<�e�<;ݫ�~���2ἔ�8<�l５�=� ��ΐ+�����t��&�������<V0t=�&��Xb`����;�2=|���6=�`���:�;^
`�c��<?��<}�м2ߖ�W�I<z������ �ۆ��܀��ˊ<ZE2�0�<�=�1�@N{��=�� =@��;w7��F<�5ٻ���=z[=��ؼ?1=;%���w ���ԼNP�;��@=���;N=�����;[;����=�'�溏=��߽�o��2QZ�e��<$o����<?�1�~TP�a��>!< �b�˽�C�=o���+�n���+=]?��j��=��(����Ό����=�ڵ��op=�s����U�����Y��s�1=R�����74ֻ��$=�/=��b�	�.={@=��)�+�>��%<��ϭ���.!�Y���?<	;���#@��=����ȽC>O=D��:CfI���K=2����2�<�݄�r�j����=&b�hA��񲽪��=VБ��.=cB�j4/�9T<��ڼ��w�r=���z����T���+�*ؽ��;7�q����<]�<x�
=l�Ѷ�J�<���J�<�/<��h��z=�悼�\����;�.�;,��<8�e�;���<��V��������y>�e�J�ٽL�	�����UU=B
�=�v"��N=G����ټaaż!D3=��<#�><���uӤ�Sp=<em���ey;�0�ܼ>�ۼ�<I�+<� ��Ty����;}�ɼ:k��4=
��Er�<''���$��]=3,9k>�%"�=I��;VN����|=$����=����p+C�眵<�)_=ô���c@=�����9�\N;M뭻z��;�#�d#����< ʠ�����@����)�<V�=�Z��<~�3<�Ph<٥~���^�R@�մ$�%7���<E=����>���G�<I����[9=nk����a`w=@��T�<�ȱ�� �^b5<{�"=�_k�R�#=w�#�P���6�o="���� =���Q3�$�=�!��'��)�+���>մ9='�X���=qv|��H����=|A7���<'��G���j��«=�)� b�=�B\��JB�tɼ.&�<8X0�Z���<�;���<�̦=��=�Ĭ��PO=d�V�J�=�|��-����䦼��m<��B;oɌ=�$������F�w<�!�����mX(�������<�#�<g�=R�n���3����;qH�=���0 Z�`�S�Y#��6�;�V�<f�M=��4=S��<J�����<��Q�e!+={Q���� �ü���X0^<P�v<-������ۉ�=��ڼi��U��<�Ž�B|���=9��<|��=�-8��K���Â<�r�֯�����=��7=SJ�<�D����3<o�o=�����4����I�s%P<��w=�ձ�����Z2=�D�<�ݔ=�Z<�w�[��L�7=�B=�i��¦<�FR��̼�3�<�8F=ܮ�<&P"��$�<8�̼�=o�=>Fý��}�b,�3�N=Vv�<ֶ�<���<��g����o;>�]l�Kdg;]����ӽ2�=��2<�ĵ=SS��K���J�g9Iz��a�$=�hx<Z��=[@ �RAx�Rt�{ F=��\�����]��]��}.<VO���@�V0��]9���
��X伓��*�C=�\e�����z����޻��=E��o��;�3p= @d��i<��<DJ�;�h���.<UV�Y����� ���<j;���=�d�<]AӻY:ܸ�������O���;=.��;�<�.�\Iؼ�7�{vm=��==oX=�}��1�<��i���Ƽ !�<�k�CqL=1�<�C	��K�<
�<#u�<lN����=�0<o=��I��2=R���]$�%�B��C�;Nz��ԓ�=�k"���<q�۽;�<�6��,��<e�N<���<�m����v���;���=4<l]=/x��B}�И½&��՛�Cc�<�P����o���	=�.=�N�G(=1�`<K��t��9̫��醻�׼�'ϼ'��;! �c��[2���<hIj�xuh��� �hᒽ�<�r�;l�c��[�]��TM�L�Z=I�<O逽}�0�H��`𼛵	���<jԬ<[E�=�Z�<�Uڼ��=��D�����;�Ԫ�-�.=�9��-S=�8�<��&����<c��g����ӻ�y�;M�� �������i�.M�=�{�;P��<�"�i�<���m�8��<��=����Z�����=�N�<����)��� �mR�<��'�����rU>�E¼F��G���U�3�W����SD�X@�=����k=�7{�������F=�=[�����Ǽ���}�t;�FM<�xj�<�N��v�=H��<��<y�o���m����q=}1=Ɇ�K�;��ۼ�%N;��<)'y��j�<������=�h+�f$��0L�����<wR=�,=%���=1��l(�����6=<\IX�A��>GV�c��	t=.N��-8��������i�t�z�<�t�< ��:����"�:��<)�"~� ��9�	=I�=:�����<�#���Bn=n<Z=];�<�_=��z�;�7:��m�r.f=诊���0=��<@g<C5ý���;�٦<;<�<��`�(3�<f2��Ğ���;���,3�=_8\�he9���;n�վ��� B�<j�@��� >/�����L�|��e/>�kc���D�a�@<���S��<'=!&�_t�w�<��U<�ֲ�����*=��<��)=Z��ލ����
��I=����)$��q�<#r�=[�=�aʼ�z�<�m�R��;�]=b���Zϼ���<l�>�5=�̚=n�<��.�y���s�@�S=uq�c�O=��c=i�8�4�Ѽ�= �7<y%)=�*׺���<�V�<��<!9�� ao9̼l�(���ѡ<���P*�b<�4��b��S?�=�>��վr�$>�ZL>��~��'���b�uǭ����#׼	Aq�y�L��,��uߌ<�#�=�=�!=�����F8<ē�=*�=:�(=�;��A�<��E�F=d!�҃'�`�<@�8<��$�L�<��;�����97�=�7��Ju�3����j�窼6�d=y�<�]=[͚���<�H=�n+���r=�l��zlλ/�p<i	�1Y�Q0=G�$=��:�<��ݼ�	=[�<��Ѽ�$=�M�.S0=�c<.�;a�)���*<�o<�g�K6f�Z=���N/=�*��|ʉ�W��.��R�3<�D�<O �<��=��V��Ҭ<v΀=�r��g�Ƽ֥�=詀=���� �9��!���kp=�����;���Ž�C�M�	=��s=3�x=�Z�;z(X:�ް<9~T���4�@ޯ��F��r���;	��=X�<�O	�H<2=�zM<��$�꼢���f���B=`�<�P�k0�(><<��N����������м��<�5�<���<�e�<���[�e;(�;��;�H=v	ҽ��A�C=G���6�=}֢��F�<|����H4=�w�s�]����<�-D=[�����=9��=�Q�X �=��*=y��=C�j���i�<v�$<�lE<!HF�.��==�����[�L=ˬQ�9c���~=8�� ͦ=��4�*�3�28����>�b	�ks�]�)<s��;zy=�$<9}^�c{u�Y#=��>=��=�<QW��͡�.�<Ѽ�<퇵���=̵=��:]:���P�;�=|<=��,=���"��s�<�Y��p�1=��<���(=�����z�< J�8���J��<�=n?�<����,�;��u�::=���:=�H=� `�u'���̻P���4˼"8N=ZLZ<6�=1ph�ˠ<#B����+=ʇ<:����@��X�<A�=���=�u�=� �{��R_���R��ǅ�΀�M�2=�f��-:>�\�=iƦ���=<��ν�ο;x�H�:�<�/w�
)P��~���(=��=)Zr����V��z�9������g�<�%=qD�<�%�=:5��&t=�?�<;����m�<._� �*<�=[���+<Y�(�s��<���<e��E�=6��9cF�h�;T��e��=�����-#��%��>�)�� P��7<<�e���<��a:��.���:����A_�1m4;�p; �=�gռ:�&=�j#<S���8��-������i=���<̰`=!տ��_?:p����!��7�<5�<�6b= -���"==aZ;4g{=�݊�z_���3<�QL���F��W)=�v������;E��>�;o}b=�k��0X=�{F<θ�;���F��X�T=��m�	��<=w�h=�yh=z�Ӽ=
{w�t"�'��	�;p`)�.��ո�<�L�=$h����]|=\�J��/"=�e<��X;	�:���<TiF=�jz��0�<�X_= ���U�<֮f��L�=�h�;D��<Z�n<)��M����<�����;4����P��m&���*=��[<�&�<:�9I=~���d�����S�ټ|/+���-=k����)�<Z��=�
�	�:ϲC<S�`=`��<`�]=y.�<��9�6ݖ����������'!�W��A��=��Y���;I��=
E���l��lB��9�U_=���<-"��=�xbW=?������<�6�"�J��֛���=4��OT=���;?�ȗ0������<cÒ�a�ͼY����;�p�=�n�8���wu=$�̻�z��n=�@�,��=#%���wt=\lx=�������=�c���Rݼ�(=���<J��:v��EmO;�^��a�Ӽ'u[���=�x�����0=~΄��<�=�=ې�����<1$�<�
�<��f�s��4�=*���=��<¦˼�B��,��=׃���p��2���]�==eμ��<y����`�=�eU=i����D���<���;�7�<�#<�eQ��iS���=�C=���i�=�-���U!<��|�<߰*=A}F��1�<.C�=4R>�Ug.�����������;^�<T(�\y��u�<��<����q��׼�3��;zkK��<Sf�=�3;����)�"<ʔ���Ԡ=�l]��U.�R���,H���@<�CM��H!=�P�=�27�����tC��r]{<��¼�X���K��Ԍ<�R��D-=P�<`�����=���=Uq=�e&=(*��x�C�����1=��R��(=���=��,:��fㇼ"�Ӽʗ��9�� YؼY��d�O�ŧ�����DZ��-�=�Ȝ�狞<�����;�|�<��>�����ƾbu>	=�;��ۡ���>��
�5�=,��<�<%Jt<׼F����=�=�I�<@ ����=��r<���:*&w��Ն���`��|ʽȣ��tJ=�,e<���������p��p]=�}���8�:�<;{�az�+�h=f  ��|ϼ��J�O����;kd�=�����>ŧ�� ��c位,Y=D5�[hF���noa��/=��G� 
:��a�L=��=U" =�a����6�擁��_p;;PW���\�S��;F]��ji��$2����< Q��t�V�w�y<u�C=�<���7�{\��f\=q<<�v�;.�½Wb<}�����z��Zۻ��
;?����[����~���=/�m����s =BX����=4:�+$=��;�Aa���=h	�� ,=�����`R=�~o�]5<7�����}���Y�w��<�<Y�E<m /�*��<�Y��0y�=�g�� }@�4��X��<ś�K$�<��;�ּu��<�g3�l=�ɛ=3��vW=��<��1=�ý�#O�<5B�m_�yPV�:��<9Y�<D��i2>;�7<{"��d����;��"=�-A���(<��P���|=Ũ=���<�� ��	��=�O<=��ӽu�=��ٺ�G=t�X=���<չ���Ӊ���W="R
��:����?׻�W��)g"���q����;��m=U!�=2>���� =���<�Pf�ö�<Ɔ���86��3�<�ҹ;�����սH*�=d��v�� #!=�-��C�	��4��=W�D=��W<w���@��p�<:�K��<|!��'x&<�@ڼuM<X�|��;�<<���<�=z�~�#=�HʼB�<rh����<43��v�����;tB5=S!���
��3�;���=��-<�)� +���ú�r��#=���*��
ٛ�¿�<���p]���Q�C�(=C�$��ȗ<�M����L�`��<�[�>��;�$=w�����	�9_-=`�J� ������T�<�n��O0����)�<� �Y躠��<oA=�h��JS�:���=��f�LM�<����R�=oH�n�I�XE���q=�_Z�!8���<A���G%=,�><Qs�<n�߽���<,�=����S���*��A�=��1<n���`�(=LZ���+�=G�{��n}��7��u�]=d �<޼�s�<-� 9���pw׽����w�L=�fQ=޻���<<��<�S<�q6���:�t�:y;�<s����$<��[=4�)=�H;)����Ah=Q���1H��gHx�ˑ�8̽�h<aq<�U����=TI켷S�:��e;ɝ���q<h<&:c�E<����;�����T#��e��@#����<�����;ȓ��O��<���<i�A���<}��4;��%�<~ ����k���R=t�>���z8=��<V�!�y(H�̕y�c�����F��Y�݌�<�K1�^�H��%=��V��<�H�<Xû�9�}1<sX�-�=M���ټ��]���<i�2��4 ����E�Qh�<ps�<�n�<���Z�<ed/�ľC<��Sy<�9	=?�=�^=8 Q<'�=���p+R���a<�rx=|B�<�L�=�wl��/��'[�cH<Ҡ���}=��a��8�����<��<��\=�c<�Y�<)q-�sA.=�ģ��o���W���<�?�%�<�dѼOa?�,;���;��Q���]�O�t���=����=������.�<�żC���g��(=��3=OB=:B�@C���#=�'=ڜ,����4�H=��k���r�  ���H�9��
=�;��.���:���,�=H����ϼ1�:]`�;)�����<��=T��<�S~<���V�ɽ�s�4� :��x<�bt�w(Y���J�2��<�1?�\�;VH�;��<��<G	�44�;�Z�<Yt�>��!�������s=
�=M7���?=%���O�=�  x��d$;� >1߽��-=�O=�|۽�=#s��7�{ʽ|�1<$D6<��N:�9���y0��7=fI%��*��N=�=���Z�c�ǿӼ�{!����
�=��Ͻ���=3���Rl7=����s�<���oM�����<m��<��=��<Q�+�p\��z-�gqf=}�="���9"�.&B��3"�����+�+�<���<��\;�H�+���䁻q�;@� ��K:��X =UxѼ��6=�̾<w�ڼ�m=ڵڼ�kл�a�Xa�;��$<������<ՠ#=,!���>���-<^�a<��E��d�<ޜ<4�����;�(=7c�i\ �%S��#<�=��g�T�=�Q��:*�*�S=�V�=v���)��S�=-3���=�h�;�>^B<��̟�$t�=�������c�=��ʷi=��5�p�G�]l;P8;�EN<L�1�k����=�^��B:���=�=����؎����ߎ<��ܼ�=~�]=���U�ʼPg�h�Լp��<�1���R<.�Ӽ�߼�`��t=.=�/A��&9����<),�<�c�;�_���<��;<��܎Y�D	=�>�=�j�;�l;B�_�j�< 4�<�5�Ϭ��f �9����4�=gᠼS�<p��<8\���5=�ɽ��܂�w��:mp=j��<:=zм/.۽�'�<R3�<C�E=������+<����-�gb�<^����%�=��Ľd�9�KK���<#l*�@T���Z=^.�8*z��2;��_��Ge�O�����<��<]�&��+:��<4�.;�	��0c[<���:?�X=A���<2sZ=���9@��J��_����O��ʉ<�s:@��=<-�澼F�:=�"!=�V;��S�<��S��0C����<�i��#�c</����+����<��y<�7��/���+=:^S9��<u��������=TC�<9�w��;�:F��=�f�<8}����<Yiּ5�<��<�;lʪ<����N�	��wr;��2<؎��� =�/�;m<\���Sw?=�����I=(;̻�K/��[�=�� �;�u�5����<��<� $=�Y=�M����*�$�=���<X���{�;�5�;)�&�]�=?��<Sk��x0�<�h|; �"���n=��~<�J׽�����º�|#����:�ej���껆������<��]�Q���@w����<�A(=B�K�Z+��9��<�'�:��<Q���㢽�>���� r�� ��u��=<=�t�½����"�}�*z�#ag�G����<��U�|��8��f=�N��O�<�ł=-��=����T5���#)=���<��u�Q�$����5�����_Z�=�<�ჽ*t��� �=%%�a�<E�B=��<_�O�m�W��z(кu�<�A�=�П�W.��rٓ<�c򺃴8=Q��=)8 �'�˽�X�C>��/�c��˕��l1���J"<���<�"�<�]���<ƒS=��=J���'[������2=V��<P	λ�+�;�9��$�� ө9���^�<�==�;ӻɈ�<�����Y��Z����^�ի�<今���<C�]��q�_[<%�2=%�9�̞j�BO=@���$����s�[;j�9�d��C[����1�u՞<�1�*;R���̼%��<TV�<3@�=JT��Լ�q
��$=G��p�+=��мO��<>0�;E�?<<�<ᔏ<�r��ӓ���M8�b��<�5�=)�k���<��m<���8\V�V�>��=9x�<@&˽�‾Uf��R>��<xG��Ly�<����ͤۼҗ����=���=��a�Vp�;� ����=O�������S����r<��$�cI��׽��p��
�==��ʼ/=@Th���<�0�I��<1+=�
;\E4�P?��+C-�(�>�W2B�5�;޽e�K#<E�q�$$T�h��=�k?�	��=��6����&�l=PK:�@H�=��<����=囑�
�z=��$��Fx=�)ѽ�7��D�<�j	=,�F��^�=�K[=j�f=(��Q_�7�r���Ńu�'�Z<�� =��P9:=���<���9�=��<��ż�����S�=M"���-G=,G:�p�<w4���Q;jp�<��<�T��һ���1��gV?>�+�=c�Y��B�$��1.�<��m=����<mM��U�='��><B	&��������ft���=�������=�˻7b�Β�;���������t�>~��}�D���]���K=��j=���K�Żoy�y9
�,���b��<Y`=R�f;�΋=3���+=�<�(�]��=EO�<����?��Zer=U���@�=ְp;s_ع��+<�@��q̻n@0���V�a�V�r*��5�2=o[��r��=�
�<��ȻHs����=�����<6f�q�)=����3_=MC���6u��&��t=�(u�
��<1���� =5��A�<��=p���O ;=rS�Q�<0o;Fz���	=$�ɼ�`�G����=V4����=⫽5*���r~�x����H=m�?�	C�=��*��؀=(8���'���q=�y�����;��g�>D/<Ԇ���� <�� ;����~�I=��&=��3��<A��<IWi���d=.4u=u���>��=|i��n�<��G���;�}����=8ļ! �<�x�=�ҡ�\����=)W��6'=��:�]����D���=wt⼵N1<*�|=C�3���j=���=Og=E!�OF�a�1�m麋�N�I�� �3<XR�����t�4���<���>ʼ0�
��&=�hɼ�V���=�K�<}ʢ�J��<��̽���<�T�=�<��<sb��R�=���@��=�ߒ��᫽j�&<�׼SO@;U�<��#=z3+=&�;�!7�L{�����<��q��C�<�>�;�����=�=X#�<�=�H=�G���r<Cd#���;��=H	<Rǻ�	�b=��ܼρ<�FO�:	4=�mܻ��=-���KA$��
��3�;��'�F=�A=�,=p�;f��:�ي�u�!���<�s��Dj=)�ν��<�)(�	��h��8k��<#�+��*��ڶ����׆r=��T�Ȱ=
ۼ�Ԣ<��̺C�N9����\�½��<t��=��<Њw�N��<j����i�0�K<��7�?F��*=�,=�S�<RN�*%��kJؽzڽy�󽓠ܽ6>ܼ4�K������H���8�UD�<��=c{�;8IU=��!�� =��;�4U=�ľ��A�<�!��N*<��O���J�<ߘT<�߰�w�%=�
[=�	6�#�<�w7�f|J����<p�=��в<_h,�P�����<�#�j�O=�����7���c���<��輰��<�qQ��,:=w����=d�$����;4���L�
=���<k֎� ����2ϼ�������<����=ł���Hذ�B;�z�<8]�����ee<K�&=�ͼ|��t�o�7D�;؅���53�X�=�,�gF�;�MҼ��4=�L�:�q=��9�I����L��mۼ�����=�p����r8���<�AK��G�<lu�<��<�(r=�&��B�<�S׽��=<�u�M�E=�۽���=�u�;t�y����<�O
=��=\���qx��D,�;
7��I�/=�J���<�*�.vW=�`&<j�8<��ǽ� m=�\�ǐ=�|��C	�=��ӽ��<|����T�=��� 8= �ܽK!�<Y�ĽJc��+=��Q=�Z�����=��=�`����/=��t�m�<%u?=[�=��酽g�:q�$���N�
,ʺ�;��+'�;G�=U������v!=�R<ӆ|�|9��l�I`���Օ<g�I�k=$��|�=|r�o9�f.=[��<��;�Z=ۄ$=�Rj���<h�G��ܐ=�o���F=?���V0=�^?<��J;w��}c��-�<�ܦ���P�̄ ����< �(<�>="��������T�;o�=��ܼ{�m=�5���w��kҽ^�M��=b�<�S���K:�1=:��^�<x>\�e5=*4������<k��<�t@=ؔN:�2��ݣ�~L�=��ż9�O�@��+����):1K$=p��:���<}T��Ҏ=u��儿<'-=E�/�XL����˽cSq=��꼳�ѹ�wp��>�쥼_��<�d���IP�:��C��=\𓽎�Z=�:0�eG��Zs���<��f��C���H�O#����M\ ���-�_ d�mr���E�85��^�/�\ �=��Y���;�
=vvؼk���
�;��,=��»���f=GO�
"�<��`=>I�)ݻH�.�d�2:Z�o�^�c�S3�u)�"Ӯ=�g�:�v�<�m�<��)�*{�<�xb�K稼[�=��):hF'���1=�-<Ι2=���J�<�U�<��<@"�:�]��6=`�;��7�4=+�3=ڮ�u��ub_��p�=<�H>Ϊ,����=�t�<��ֽ�揽����7��<.�����=�`�;u/�=Gam�A6�F��=��A���x�O�:�[�<��(��i���Լ2p���8�<�r<�#=R=�����)��\<�Ԗ�6~�҄<�b/<��A����<�s��B�x�z=p,#�v�8=d����<g�v�W��(W��x짽��<���=�N|��3����-<��d�8�<&�/�g��=E+�1a���YO<��޺,�7=_: �Q�U��b��2�=�@)��� y<:W���Z��Y�<o�V$�<���3s�[q뼑i�����;�j|=7A;<��|=���<�s����=����`�鼡����=fی:�]:��8(�21��ƥ}�[o��%��W�h��"�F�G�'�(��u=�=��9��=��=Iav<�Z�'m�N_�<�x��}�<��<�,?��X�d�=Z@�1H���<�R˼����=Fր=�`&��Y>���<v�3<}���>ӽ��LػV�7�����n�����9=�M=J������ý�Mq�և�=n�Q<��/=c��[�y;z:K���B��z;�,��=��<�=``f;��<�O��h齞�<q2�:�=�=���;U9W��������a�<�0�b>�О<k��<gm�;�q������Ą0>���堎=�v����+���D�4��Bo;Z�k��-#��r<6@���(<=d�cC(��;�!� =�JY=D%{��C�_�̙���=b�=��c�����HI��.�����p܆<��{��=��Z�<NR�]�<���[�=�Ƥ<�ܰ��y�<'p2=j�u�/7��I���e=���۴�kH$=��=yy=���C�Z=�pѻ�	�77���7�����$�����<8�V�@�<�ũ��S��b���o<[��8�t�M�<���=�%;-���1�=�{��Ï�jQ=�nn��ؼ�ނ��R<=��=��=�S�=�F������U=����)�z颼%�=E���Rt�=4y���O=�Z�=�c<̔Ž�A�=����Ry��#L=-|�<0^<���A;=�R�G})���N=���X��W�<�
$���!=^�	=��,<O1&<$!t=f�K��׍����=iZ"���<r�l�r�Ǽ5L=E�ۼ���,:��,�<�;�2λ����3�޼}�=@W��^�<������<u���C�<.��<s��<={t;���/�k��;�T+=��%J[=�������<�2�;�t2�Pwr�\�=Ų�'�<�|?����=����`<د[=����whؼ�Qx�����]��:m'�VP=�/<��R=~��=Wu��j�=X��k0}��=�3)=��Q=���<������<BH�<2c=^o.�o<2r�<��ۼ:�=s�A�4Z=����F��=��9�염�$5��e(�p2���ϼ�� =`k�<��{<��l=�>��\�>�<a���1t�=xy=<��#=Ы���<λu�@=6�N��м&��<�"�CR<E@.�ǥ6�B%ż�i;����)Rs���<�= K�����<�@+=b�<l<:=A�*<���:+�ټ�<=eׁ��z�;�k|<�T�.<�;�A{=}7�=���<-
����p�Ȃ��.�k=\���1�F=:�K�\B��!�g=46=��s�3�O=g�����Z=�@x��\�<���<x��<#!Q=��;�a?��<��=#�=���:�ߕ<��:�n;�a; �ϯ���Z=V� ������<򫳼�=�=Ŝ����ɽh�}=^�%� Έ�w��Q�S<�>cz���P&��μ>���p����a=��$��n:=�_�:��0�Y�[;p(�U�<���{<�O+=���<*1-��h��֕,��<���PhK=EӬ<��A��Ӱ��S�=֖�?\(���?=����ፉ=�<;�Tٽ������=���qJ%�K<l�<=d���ս/�L#���b=�i=L�ļ��񽿝�<����&<�;=J�e��]�9x+.=y����F<ƀ]=O�+�����aX�=���;(�=�%����<��<���!`���m�� �q: �	<�<�)=�p̼v��<����I���P��4=�3��+��ǻ:�ľ��?�|��@/�< '�<AȮ=œ�<�C�M�=Ex<홽=(������3~)=g�"��͕��F=��Y��E=u�t�%��^���X=skB�ew=#���/��K
;��<�����˼��~������Ne=��q��H=/���d�<���?�<d�L�J�F<�򐻀����(�8<˳v�=�<��<�n;K�=��=��˽*��Q������"WW��G%>4�9�-�s<~R��B�>�=�ҡ�ɗ=�v9���H;]ا���;;�s<\�=�{�T+���P=�F�آ=u�<�t���{�v��<p���6��;�lH=P�=���!9��.�M�G���EN��[m�w�=��g��t#�<�+�8�����=�z<�J=��]=lm6��!�<b͇�=��	=: �<���<��⽿�k<酼Jzv=�zX��R< \e=i&���;�u�������̾<��7���<~ゼy-���Ӵ<h���U�����<�Y<a6��2�
<-a��\)=���;1ݦ;�n�=	_�<�yû(/���=�5<�	�<�? ���Q<��u;��Ҽq}���֟���<%�=�i��Z'<\ES=5��<�V<&Vv:�<�X��*z;�&�U��9!����=o�<��0<����b�;�5�<)���G�=�3C=�@h�e�=j�&uC<��.�1�����|<@'�<��=��Z�.�»�u�	U�������=.���->:5���h�x=h#�<n�J��v�;8�5;De޼f��=ӿ< .ƻhH<&�><��<�";� 
��X�<�ļ�ͼ</˭��<A=�t<�XO=��}=w?�<�a=Z�y�ᾚ=��a��R@��G���<cn�E�<���� ���;�n�A=[�=�x+=��9�O��Sg=����-��ed<���<�E�<X��<��W<{����9<r凼a[�=�I�<�1��SA���+>��=����B�<K���J��w8����=K ���C��m=PeU��hN<ʒ�'�@=Cӻﭚ=G�9����=?��<}����To��9(���ݼ�z=��;2��=��A�J�-=�D��o=������X�l4�;xI��Q�m���o�]�<�T�<�MA<���=�<���A�Z���sS;��A�%L�<��g��G߻*�:&J9�3�z;͉==��}�<��>���d�����k=03Z<lc <�U<�=�1�<ll'�d�<�����G���U<���=A���x�<I_w��s�]<�,���R�<.��=0�F�ob�;yb�=�x����(=P����<k0�<��A��k<��y�E�"=�#(=�<<
����==^?�4�9=�Y��L#����^�h8�<T��;NP�=d��)��<O���N<	=Z�<�А=�h��/|�񢌼��Y��+%<Q�.�y93=D�r���!=?�G=��.��=gq�Ic�<'V<tB�=ܗ�
�½���<��F=uD
�T?=��W<����;��\���<#V��=���.Lm=�oV���0��a1�M���<��W�D=�%6�#!S=�����c�j#�Y�=��@<�p��J� =�6�C_��6q3�
b<�����ӈb=���<�*�F�d<<k�<5�,=]dX�H	>�O��d���{:���͏�t<d�u�3�E|������==�ݝ<�q+=TO?=�L�<�z:w���@=B�����:^��=n���R窼-�ӼGz=��
�T�<}�b��$=:��;f{�<U�<�E׺�I¼j�I<�T��I['��/�9�<�&<胼��^<Td;�u����<
bl�#.���6F=Ќ�:7��;6V�<N�}��2�)�ѹ��=wIf���;�F�Ph#7�!ἒ��<���;yk=�D<�W�;C�R�=�؇<��廋J��{�-;>���<(��3�������;�=<�(=V������<L�g=�-����</�����<{DJ;J@(��<ż��=��!����=L1�|���'=#�=� 0�G���Q=��4<�ϰ��E���3�O�=_Aӻ-�=e�$�1^8�3}�;���r�w�뼮:�n�:�=�%��v�8<#��<Rf=4,�,��ݴ=Qg�<ǘ=���=�4����)<8�`<�y<��;�䄼�=w�׼?/�Mo.=l�:��g=T��:h[=�S?�;~s��99�\�
=�W�<}#M����;ք���1=c��lڗ=��8`Ey=9[!>}��;�T�n�$=�ɓ�j���oܽs[��d���6�u;�w���^��:#=�xr=��ݼ�!�<����<ܿ^�׀?�<r��=�M�<�,=,(u�(�<���� <=��=h�_�<f�q���c=b��+��:���<g�0�����P=��d���b��Ҽ?G�e+= ��<L<<��~��F��� ��-��;��;�JP�d[1=^���4�?=ʮ��ǰ��v-�y�4=�< =�V��D��o���4y��,=��<����8�6�X<���=�h�<�QH�z�=��{�ԉ5��MM��x�<݆=_#�;�X-=Ϭ|�[=������<�z;��<��=�&=C�<�ϯ��H�=��j8&=O�<����_�=E��Wƽ#����W���=N�ݻ�M�=O@C������W>t�,�l;�� �8V;�֩<+�Ҽ�Gü�Е=:�`��<I�ͼ?�ļ���;i*��'��?<.��<�b���`�<j2ٺ9�һ��=i��;u����4�� \�E_�=�4���r���I=�4 �Ű��[�����m��:�ޛ=NVY�#��;��:C33�s_�<z�h��6�;��\=�cH=�wo��7L��d=5�T��R1�< ܐ9�м����SP"=R�*�%�Ҽ�[��,�Z<q�3=JrI���T=ZP=����<W�9����C;���j��<:ɑ=)=z��Cۻx�꼯�8<��=�焽,)�=�W��Z����G=��� �;e�=c�J=ct=��G�H�::1��1rk������y>=�s;���&ڼ���<�������<��<��0�+K�v�H���"��$��P>F=��¼K=T��=~�C:�9�)N�<T�:꽠>�<��U=`��>�e�<�=C<S%���L)<��<�7߻��%�p�=��o�]��<��D=ް�<�X��E,�`����4=4�6;�O̼�꼗Z��#=��}�Y��g{<�0���S����<� x��IW=��<��=�V�L���܋�V3���­�R��=F7��͑�<���=7ٰ<_!�<�� =�倽�bԻ��%��=��=��m�7�
=~�/=�g#���̽�fX=d�M�xn!�~�=�3W=<Q�y:ՃB�V);�'=Ů��G�f<Xq=�`�<�8��������;=j����ż�j����׺l�@=�γ<|A,=�<"�8�7�#��;,
�<�Z��d��eL\�|qɼb+�K�,�+>�>н�#Z�0�>u�=uB�<����<�vۼ��<��<�%��}�ռ��?= [�<*��;��<0��;ɶr��_����;Vb��y��*^=4(g���*�pb�<)�;G6񼘈��h�>�*Ƚ��Z��G2>3,o�僀�XM�%Ӽ���<_��;��<�ɪ<+�)�bN�<�)u���,=�t<�~�=�ka�Q�
���k�<�),���=UN��p��ҝ�<+-=_�*�4�k=r�z�f�)��'�=��(=]���H�Y��=*� �x��;�=<}
"���<���<;���2;�$=۳���ˎ=iVc��3)��:�<��H�j��;6�<��=G�,�U����,�;!$��o�=�����"�<��2=��<<�[��el<	�.����=:���+`�5�<5Z�=��<��5��y�<�=]�����<?&�4�'�lZ��X�N<���+�=h_üZς���'�%�8�*Z��"^=�{�<��&�2vU<g�#<oX�<1}�<.<=�]�<h`���A��떻Y�;2v���đ=OK��H�����<H��=�_�<�����t�H(�Ҁ<�k_�"����|��*#*�V�=86<=��ϻ��%=XZN��nB��b��M	h�|�A��$:�����r=�iȼ�ɼU�-�|a�G�<�%�=���;uڬ�a�ۼ�5/�;�R��P��p=G�=鼼5.;����*ռ��}=+d�<1��<ll=�]P�A<�o��c��<�h=&>�Ԓ=FR���Kh�`+=���;e���z���=#��]2���Z =�Q&��1%<Y�9�=�������-U�����='!��׵�)df��ե�3{�<E��,/��7=1�8=U<��=ԏ�<՟8=QN#�kD<��v=��뼓i��Er���B�<�C >y�JzK<���<��ڻr��<���;,E���ۆ<p�:Z{!=�"<+kB=s�r��욽����R��<��=��=�<���m�r
=O/��#�3�=2:�T��Ax2�-h�Z�=��=EҼ��� գ��w%=���ﯽt��l=}�b=�Ғ���j��02=�}�O&=.������P�'��� <y4�<�f׼���� =Z��<媬�/I�<m�>=#&�IX�=��Ѽ���;�r<��(�1��'^m;�X�<G����޽��ɺ=*2�;�-�<9μ%M��M�'����<��<�K���g�=)���9�=Q9�� ���M׼��:<{�� >�O��= \��0ۃ�N����">=^��:�O=?�H� ��7����@<I&=��ּ��y����w�<�sH=�����=�ۻqH�;@o�<T�8��Ȝ�4��;�,ۼgv�<X�B���=�%ĽF	�t�ؽ0��:<D@<Z��=5yM����=
�伉�%��
��� =z=b"�< A,<0= ;��Q�m�d=����8�<�o��sV�㊞�(�A<�O��a�= �۾�<����d�=x1<�3�<O~�B}
�(�3��ˀ�Z�S�r_�<*��iz���ԧ*<'�!=�<]�=_񹼯��>���]~����=;\=�@/�G�.�I���T�<F=8�?�́L=DQ����>+�<'"��]�=�6ؼ�!�-̓�c̀��-��� ;�h?=2���b!����*;�!���/�"}��A=�?$�,��<�`��l=θ=�(=�(��$��<�� ���� �y<�-=�D�=Siܼ<�*�C��<�z�<=Ѻ;z��������=a���(=ٳ�<s4���ֈ<W�=�B��� &=�)���S�8\�=]�R��"�=��m=�VI<�Ÿ����=���;�-�.�9\<f�ػ��6�~�K��l%�	؝=�=�L�;0s!���=��Ѽ���l�4<䌣�DJ�=���������v=c��<��!�PDĻ(�=F�<�4:����8�9?�9=Ŝ����1{�����h�>�D�=���!�!��e��"���E'��ܽ�Y#�|2���V=�[�����X4����<rY��o=�\>�	���� ��Z#�.����̝�2��< ���]�<�=�Y�<���;�C<�¼�����ڻn��<C���ޤ<�O��]��l��o��9Q<��=n'4��=�d����.`:�[H�;���Ԋ�=s/W��=�)=
B�;��u���<�Ek�r��0X�;H$�<{"�W5��yټ��<N�9� M����j=@�m<+=i ���o<��=\)=��=����wA���;��:�؆G�.�W<�{����G="�+�V���һ>̐<�§�xi�=m�+�a�<�Q=��Y��uF=Vd��B��;V�"�J��Hn5=�]+��؄=�t���(��rP=ڛ��d���~��|�;�E��6��.֏��ّ���=��;�L=��<?� =|��;Xv�7M�	=��s0i���<�S��/�S<�O\�5A�<�򔼚FJ=D����7���q=���[�r=���DX��&��<+�g�:�=l��s����v��=��i�D=.��z"t;i������l�g�'[ۺ[���[=��;�གRN>��R�)~�DWZ=f8��R/�9G�<T<���z�����3a�<��kNj�/�=��E��B�$��f=�6�v#o=&����n���E<c|��ICϻ�4<�fک��;>v="��<)��t.��u��Q"=�m8�'.w�1.C= ��t
�����=�@��&�̦�<�<x����<�U	=�<�`�<nՏ�y�3<3ݼs������<��=V��<�S��3=p���V�<�r��@���;��μ�� =դ��r��P�n�n&��FX�<���<|<�ټ�rҼa�A��ZJ<|�<�<=B�^��c;v=CN%�.�&�
V�h�H=nڼ�7�<.���$:d<�;���������Ê��=�=:�R��Q={`����ۼ'P+������c=/���#��Q�[����<T��K�#=\�@��<�<>7���
=�]J=���9D�<}�=j�,<��c=]}V�UB:�W=����������<���<�2#=��<�Sns���z�=	����=��@J�˼��=|y�<6�	�U��IV��l��󑊼�=�<��<�����<'�b<B�f���+<YQ��$��ڮ1�j��<R+b�d-�8���x_����<��<޲<d�伕�����="��=#�-����=��-=c�JV=�r޽95�����8=�@�=���m�=��=u���2=�_�p�!��o��������<�=�B�����<�O��4z=��z����ʹ��	���?��uxU��9�T��<���_����H\;<5	���+��(=�=p���W��/G�<���T�g�W���޼	�=��`=���<���})���A�����B!���<3N���<���If�|=tO+<2��<��t<6�w��={�<A1+��C�;p�����8������(=!�.=��<|�Ƽ�D��SƼv='<@�ؼ�;������<�ȑ;��<����C�9���=5�=A�_���!��<�Z �v)<߬�B�w<�����8���<���<��o=��;0�	�.�<9ּݿ�<�w���_0=������5a=�X?��d�<�8�;�vG��W�Na@��j�ԍ���wz<�~"=to�=��Q�̼�Å��h�,C�� �%�u�R�\��<p�=X����(�=�O�=in=ly�c�.�0�=b6��-�=��<'=.A����<E	>w��9�EK�<L0<!ţ�U|<��"<���;�lF<1_¼����N�L=�����T*<F�<;�f��<=$�<W��Ǫ=_�:�^_�k�}<H���㝸=��<��*����"b��w���r� �B�z:�ɝ=<�=��R�6�����+�@9��8���\�����=��}���N;B�/��	���=�[T<5>�1K=0����
�<�\�<RS�ɒ�=��S��]Ǽ�y.=����';M;$=`�z�t�+�0/�;:��<�Lϼl�<ux1=�~4��n����)<�4u��q8�.�:{�;�j=���o�<ݷg<�$M���뺌�;:#H=�����|r=���	j*������"����9:��=����� ͼ��>Q���,I�=�I佇�U� ��<�\��^e=�i�<���;�s���8U�s�=SL��/�X�˱�=�멽��:�ܼ"��=
W�:Y�;��2=ʟ<>Yb���ؼ�򄽥�t<�)8<`@H<
9=>mq��,�<3��	�����<�x���Ʉ=�Lf�����F�<7c!�ls�;� <�Ĺ��hR<�eO��*��(M���=�啻��;c7=I�E=�Ī:�ꄼF�o����_�;��@���&=�p���1�u��=1J��񪼬�e=0Y/���9��<8����ق<8`=" o;L4-��/	=��λ��ὑ0��V�ٽ9�V�8t��IrZ:�Ҽ��;�s=!�7�[�N=9�p���<y"�=�H�9w�;�q��Z�����<J���)��\�e�Ȩ��$������*��:MT=�i_��"޼K1⼙1�<λ=��<
)v<����/�μ�Ȱ;B��E|<=�U��W�<��<6�y���=|_��C��moR<�s���w����}�F<�%���w=��¤=��3��;��]s=z�k<���<g�=��7=c�8������<��8�<���wǔ�h����򜻿B�<�K<H%;V'=�$��A�;�<0�ⶉ<L�<�y<��)=�d�=��;^|�Nc=ѕ�l�����<���t��)];r1*�y��x�-��;��=�Y;Ɉ=N���Xg��]�y�=��<ر�6b�<��.��M������a��GI�<[=o��;)��;0 =j]��b�;�n�X������=�,G�̰�=��=:��q�m��E����!�%��ҼL:�a���=Ǣ�?ј=@���.,3�5�==bnƼI��p 7=�楽aK�;ó��J��U����Uf=������=<�����=S����7���RO=�;Q�g��<��E<7*��P=E=��<~^��e=��=��]��B�:�3:0!�]������,B�=i�y���B=�A�����,�=��L��:[ �=���FXz=��=�R�=����^�<2_��ʨ;nv��lnD�񥟽���qH���̝=��<]z����=��<a)�)M���(��i�<����,��s:ds_=GQ1;��<{)W��0;bQ<RO�<Jz�<���;n�<̻ �D�<�s�X��;���`��;ŷ9=9ｽ@�C�쎻��z�몂;�e<q�]� 6%���"<7��;0�<��J=^��0Ѝ=�a�������ջA�� �?�K=��H�k�;5���'�^����0=.1��ŉ<�2�;|NW<� <G���,��S�����)�=�$e=�@��4v�<א��WR��b+� �3���:� ��lܼ3Sɼ�B�=�������R������w��=≸<z5+=y����<;ꧼ�~r�᥽�:��:��>N��=ٹ��A�+�*�>���=�A����=��N�n;����@^���G=����S��X�<
��?�C��C�;���^��]v���y�<z�����;}==W`�<�=.�� %��
�ov0<f����;>s�<�0�;��;N*�'C��Ā=���=]�����;�ռ����Ѣk=\ͦ��Cx�y	w=�h�e	<���
�c;�Ţ<M%=�k��ц�jL;;vL��	��;=;��;�00=D<�;:��_�7���=�l��ߙ �`��:{.P��-p���U�r���B�¡��T�h�Oʊ=�4 =��l�I�=&!�<�Tr��Ǧ�Z����9I����<ҷB�H�W|ün��</U�=��I�j�o�O���+<�/�N{2��<_�I���A=ʳ�<q,�9�0R�&�<0ъ��|`<k6�<_[;��q��=Q'=qK$=\=	�;ž&<�I0=�y�:�F=H��<�L!=L�����=���|��< nC���8��G��!������s�L�7=T�<��m�}=g�;�*=����>�7�*��=A
;ƕ�<s!"�1�����&;_�<[����?�d���<H�=^4�;��<���< �E��>C�C��Ni=�f��#zr<GX��p��̝��='sҼL;�8�r�*U������Vu<��T=Mz�<�@{�󋎽e�M����R����'=ȸ3�}����6��J���Y�-=�������3<�����ڼ���=�˙���<�0�K�_���<����B��<p5=2���Q�=�����^G=��u�+|V=�E=dub<��1�ٻ�H='�\��Ӑ�*==�����$A=f��zT��dd»�{�F?�J*�<�=뼄�M�v���@=l��<YB�<ӮH������cC=�>>=��H��ط�J[��<{<�1C=�<�<o,�ٕ׼Yb�<dm$���X=�KS�Etb=��}��_u�Dq����"��j�<�׼H���yd=h�f��&=��;\@ �ך �`Hh<SF��� L�O�=��g=��/<;t�~Z�=R��%��C;/Ӄ��3��V`����=�������<�5���Z��tO�Ȏ=h-���=V����;ڪ)=`�==s����,=�n�x�=$C���	=�����=謧<���=�1þ!ύ�cQl���<�%�;�S��@}=��_���p�T�<M��<��y�A�*���༚�Ƽ��'<��=n��¤a<b� �|A<�l.=��ټH��<��-����=���,S���k�=VŤ;��׽�\2=5���m��=vq�<��N=�&g<�JN<���Q
=�yw=�}��;�μ�����<������@ ��ͩ�#�%=����E�<�ɹ =�q~��
%������U���P�9�⻭��U�< �)����<�E>�>�Y�=B�=��v<?��g:��U�<�~���0�y��<fG=�<�3<��fT����<�a�;!��<�G6��j�<������<�&����ǼvEa=Auq=��������S���X<��]:l�@�B$��?ɀ���=;�p���O��4E=2�����<��3�0[G=bQ<4�J=���; G��24�E`�Y3�<�����<�Ż,<}(�;��#=���Ne�<Gq>�Uh�<�� =~Yڼ��p:~�;�J�;e�'=R�O=r����,�{���8���=�i%Ϻ����u�;�
�f���R�<�9=6�ʼ.0�<�Q�=�K�!��/���=!=}���=��O���@6=
�@�Q<41=R�=���D��� <�?��O�V�� =}d<����-��C��<E�2���:�`=g���8s��4�=[�;�my=�)=�XE<��߻Q�=C*<�@�<+-�>�-�_Z�7&��ͫ�HP=Ҙ<<G�;�tW=�#��g����[^���;�'�<0�uż+a=G�)=��a�Q
���z��o�����=��@���=��@��?T=�R��&k�=T�E���{=L��A12>4N�^��.�c;Tv>j仆�=���>��H;1��%=[�c�=H��|�;��>��8��ċ����|Ǽ��;+Ž�M�=�e����=Yc���}i=n*����=����c;�j6;v� ���8�d���`}2;�դ<�J����:UJ�;�����㸼�=.f"����<<�E�U����bռMYͼw_k�>7�=s<��߼��p=�74��)!=��	��� �Vq<�]_�N�ڼH�/�����%=��u�<]������+9l��=�@��p���.���=ɐ~=��^<��4��;�h=D^=�<=���;1K;4��;�F<�U=���M��X������>!
��ͼ=����X3=�Z꼮8}�
�=G��:/��Z�C>��\��=�eڻ�׼��Z�5ȹ��B�;�v���T��ټյ==v��<ϴn���<�P=\ޕ������ʼ�	_�2⚽;�g��a=�U�<�j4=�� <$�;s2�a=���=�̽��ͽ=�&<h4���=V�=�ً<̻�<:�(��7b=���4y��^p=� ��H�<(�:^�{��<E��0�O<���<zկ<�a�C�üh�j����<>#�<��94��������x<=<�\����R�n�<��]�b�~�~����#=�C<���Ɔ={�<�d�.�=����	�;�����ZL����=?d=N{���kj=���;�Q=�!�lAܺ�Oo��Z�<9����}<��z�t=u1\=d��;YXu��_`<P����}W�����9��˺�P<��<�q
=t����"�<��?t����:%��<��q=��<�l=Ѝ�;��6;hW����<'���k����=L�<��׼�Ӽ|_|;��ܼp�l�~��;�O�=�`=Lc�:�+=ec�&6,=gyB<s��z�����,=Q��>F6��d�<�(@;���;��<�'<��=�=� �����Px�;�я�~�'����<kp����S=j=FR��pw
��(�;q��<uL��=�NO�ǳ�;e�`;=!�T<�Ă��=k<{ȟ��l<�A��	=��u��~7��2��2�<$>�<u$�$ȼ65�*:��5D��Q�,;:��捻�t�>,���=
�.�@�B=ݑ�;�P��{_�}<0���㼓�=�]Z=�x=��<��c<�b�=��= �6u�UI���=�!��>I|�<�8�=W���=�t�=�Q��9���!�6w��k'���w;�ω<Sd�<��<�+B�K&��	=����7R=�|=\���	.�{Ќ<�v�d�����T<1{��>�Ľ��>}}���C����A���.=�얻St�פb<s�>�l�=�q�:I=7=qF)��s*;seҹ#E���]@9�b{=����='��<Pg��?�>�2�<1��f9=��=�m�<�>�<�e�L���#=v7����g<{���T���D�;��+��#�; .=t�
����<�]=�ξ;�c�d�7��� ���<|zC�۬������\��<�$�<U��~(K�tV^=��$�. P�S�=V���?����=1�R�w:=����+�A�ͶU���"=�<'(���A�<�s<ip��H�<&�X��B�=:�s��I�;P^�(|7�t��a�}=�w3;���=^]�<[�</d���'=�Ƽ��N�Q���6<��96p=��T�?k�c[�=@�<S����\�Hcc=�I�:��<� �=�+���N���r��Ґ��b�=:���:u5< 3Ӽ����O?=�EI=����q&��e`�<���=*�Ӽ����鞽��<��'=�g.��f�����<H�!�#W@�C�e�WJ���q�{�`��="��=�7����>=�ؤ<��y</�5<4|�ŭ<)�0��v.��{����{�;���<�g�=Ta��(���{d ��L�;䭺�1=`=�ס�7��<�k=E�$�_6�<�?��뽼���<��J��t;G��;�:��܅=�ۼ}�&�{�����t�l�=\ѥ<R�<�<�8�;�K+�^�;� =��4=���<_wǼ�=�(=<��<_6�;Qbw<A4��[Ь<����Z	�@{=���2�f=�}�<pS��>���XuĽVi��vѼ	jϻv��=n�����~
=���<�-Z��V'��?=�>~;˩������x��O�3�IB�<d<��w�!�����h<}fԼ����<<�Ӽ��<2�m�Ŷ=E�<m�����=Pe�b[/���D=�L�����/�=Q��"� f��eI)� m=�� ��TA=��s��b=�Av�
�
����<��弒9�:�R��N���t���Ҽ�a=.�����=�=(*�����<	�-�տ�<�zQ<?��Q;=0M]�� %�F�=k�=�V�y���<`d=�����o
<k��<C2�<[�Y=
�Y����:i�r<�����<Ӳ��H�=������]�'@i������<��#Ӏ=5\s<�u�=~m�RG�=���<%H�:�=8B����=n�4<t
<��=��,�o>?=�������<K�%=֯��uM�;�_����Ds]=�����=�/������ս�D�7.|��LM=Ƶ����)=��o����;�;�<{�M�!��G��<(^p��Oڻ!�;��<0V <�@���ݡ�w����ݼ.ʲ����L�;"�5=d쌼�-=)�ż��=��/=��������������7�g��j��<be�<xR������=P�;��<�5��i�;kx,�5@�y�������6q�=�1���0<�6=g�-����=3��c�����;� u=u�<3J^=%�CL�����k�1� ��b�;t�
=dl.��x<�G=R1�	)�<�+L=�,D��Y<�4	��t�=䟣���@=hܗ;�������L�<yX½� �:���<��%��Җ<hR<�������+�<�ݶ;�C<����KG���=�@<S�,��n%<��,<��/;��<t�����)�G��r&>3�n�f,��@=�<��<Z�<���˧X=�S;v�W�K�	=�Q�;����LE<F�P=|��]���㯦�A�T�Ѵ<,�G=b!�N3�0s��`����5?�f�E���==�����85C���<q�ļ`�d�/`<�$�<���:�H�;�s����.A>`p�<"$�����ֽ�����V��R�\=�¨;P�K<x$d=��	j
�d~�<�o*=�D������P]�<%�WZ�<�2�!L�4����8��<�V�:wBA�)���b�<v��c��D҆�� ��)=�-�'n �`�>��-�<�
��:<�z==߼��=L�ļ�=㼀����G�P(F�î����<>�/<"S���<\}0=Х� V <z_�<�sL��?-<�ݓ�<�6<�ڝ;�P��
=��k�ϩ<w>�5y��d��]E���>�U�񼧻q�fT<-D=�ZT<%�[;�	�=�*#�%�ƼG�(�*�_�U��;�z<��`�~��A�<6�-��Q;�H�=��,<G8���K<����l%?=C�<f�q=��_<�s�<H�=�m��ñ����o�<;�\=���3:J=�<� ��@�5'�\I׻;�P;�e�;��1�/��B�1<�v�;.�:=�jݻ�DB��rb�D�9=�����W;��=99�<'��<P�P��%�=��`=<��;�ls��[<=��4=ZgL;������=C*��T�=_n�<`�j�w?��y�A=#7��A=�<�/���9�=��<���=����Y�U�N��Q�����Ro��<rQ;�S��a=�9��a<�P1=t3���N=8O����<PQ����=�٫��'�;���<ջ�1�Ľꧻp�����Q=d� ���$="Ɉ:L�-;Z��;v1�<+h<L���܀;Q&ͼ�^�<ira� 	=���;���=��=�7.<7�f=�=�<��4=�ju=��H:%[����Z<䆬�� �<�<����ּ��q����C2ͼ���;P'�;x�����	���ɼ�3/��*=��p<���;���:?�=,�漑0�:߇��)K=1���ŝ���b=��-������H%=���:'{G��ݞ<������<}V�����`��+���5i�-�����V=�
�<�=����Ȼ\�R���3=�X��(��&����j=�v��\)�;_[��������M=,�=a�!����<�{=��׼��=c��e׽Q��<E܎��>�)=R�;a�=;^��0��<K���&F=�U��H�lW�<s�x����~�s<fxF=��n=����lZ�<(����;�mȹ7�;=�w�����<>�==^�s=�0��VF�
.3������2=Fl��P1��G�<G+=��'���I�Ƽ�"<��*�7,=�y�;˅y�$�
<-�$�c�R��Cϼ�G< ӻ���;�=����,�$����=h�>$j_��i>�[��G`=f)B�:�=43��ݩ:�s;c;��������/����<H0�1=��X,=�^r<}=Ǽ�)=*~E=t��!Ԟ<V=�:��N<#[/�v�r��<aG�;�=	�����̼#B*;�-\=��W=��;��Z=�U�t! =��a���s�5�,�sIĻ6�:PȽ̹�����==Y�T��?b=Xl��៼KJD=�.�=�<g�2�V籼j���>Q=�aw�<����r���Lغo1�X,�;�}={�<T���C�M<���;0�%��K����U<ba����5�`-==�0�:E=E�*=���:��\:z�A=�����F�g���)���<��R;��/�g̋��I���0p=��r<�&s=Z��;΢�<Oc�<5-��?��P�9�DA�Ou�;�C�<O��<����[=l����<�Ei�gؼW�<����l��֩�;X���<=!U=�F��h�<Ĉ<�\<�r�<0�=��<�>C�QEż�'�:y�u�� ��q�<]<��X:�ȋ�� ��}�ٽ���= �g=�	����>�`P��)k����=�lH��F�<	Cy�L%%=R�ϼ��/=��r<�#�=VK���7;��=3��In����=ّ����.����<�j��k껽=-n�gl	��SU<�O7�S�^=�,==8!�m<A�܉����I�v`a=�!���Ĕ<f]<ߙ����1<|��<dk�ܼ0��<�����T�T@�����Y�<Gｻ4v9��gj����=��"=�\ <��b=~<����%��<���<	��2%�@�w����SY=2�`��4=��<f>�<:��<H�l�����)=<4��=*�	�	=�ἑ��<�I+=@8&��&=>��#,p����;_vE=t�c�213<J�;я�J�j��v���u������<v�<X��`�H=wa���jr�O���=Ibx=!��=���h��4h��|�;Y!ؼ. �o#��bB=ܓ=�?<�ݼGE7=:E=���������4�<���4d ����=�lA��h����=����͔��'�H��gh�<r�`�A]��_� ��|<��7��卽9�P=���jm(<�C<�X���:+�Cڈ<��m=C��<��j����M�b�']=�h�;���9�A=�0�<��{�f�a=�q	��`���<q�i-�<$�:Yu�)��=�[R�#�=Z�=<m��kZ�[���=������<o&��ϼ�<I8�xM�;�{�<����
���I-�B�r=3����}J<�x~=�<Ca	�~�J=\ѥ�u����T���q��:����A�A���C=��(=�,�lp==s=��N=�L���T=(��6��<<7�K<��)�5�Y8�=ĳ<��=%hż ��<�#C����=�9<,K��+Nj<�׆<fq={����\=\�<����<r뀽�\���<��y=F�P<���ɪ�=R+6���<!=={�k�O%���=�4��OH=ݔ&�R�ٻߩ�<aT��(�<FG�|e��:y;���<�ѻzk�<'�<��<�ͻC�S=�P=P���|<]�=�<�!��
�<�<Z�|<�ۃ=�H���=�h�<
_@=E�*�ۄ���q�!ļ���<a>���u=��/<���2D�Ju=Q�3�dGؼ�#=Ú5��l�<)~'�v�=!P�<�"��T%=1���]�4;���<��4=g�G�p����q%L=>=����X�=����+"�<��x;1��=g�<��b����<��P��\<��ؼ$��;1zb;3�=��߽�녽Ӝ�<N�3�i쀻"Z�;�%;=���=�R�Z�=ۇ��{�p�=���<�'E=f��<����ז����W�?�h�<����S�"=5���kt9��HL;�Ҽ�=â�<�|y����`�}��=�ݽ�ء�i�_�HH:���=6�<Y�"��<(�f=�e��B ;��<<��H�2Wo�8�;h�]�?E�;�O�$��9i��<�g����<�Ü� ����&_=��d<H&B<��<�� =_�*<�sH�CZ��|�����W d��C���v
� ����e<(��;�����K�</�=�Ǟ<
2�<0�;)X=V�<��t��a%=ܠ�<��<�Æ;��I���<wy�<i�2����=|h�<�(�K
�=O��<�B;u}���ٍ<Gߡ<.�c=�m�<L�<�����P��=�#�k�<Ry=v��<=�漡� ��';��9�½;L�B=���j�%��'=D�*�U
�ZaR�����D:2|=�Ҽ�j�<��!����<p���\����$=�w=�#�<��V�K�D�k��#?=C�ݼ;$=pf�7:�=�
@�̱�=4Q7=o3�*HԽ~ϲ=!J,�Ԧ�<
�"�4yQ��C��RW=��,(p=��ȼ��=<&<�=]?���Xl<O!�R�]�b
A�培�3ʉ����<���<m��<z��<*7�A�<������켲�H�l�ֻ8�+=iX<\'���[ν�D�<�-=B�h=��<��1��Ͳ�T׏�`���
�=�i<���<,��<��
=A�.<�"�<�N=���݄0��Q���<@or�x钼�)�<-0#�Fr�9
��V+�D�U=3����[n�*�ļW�y=��<��J=$�H��{=�<S���a"=E�ռ�d�<��,�A�޻�F0�&�;k��<P��<͊�<���<,� +I=�.�<���:RՆ=�ʹ�����&8��'=ﶗ=�*=A�#=8�M<"j=�	���C=J�;��%���<�5=��Y�o��<t���c�T&���c;����6�

!=|Uڼ�9�;[=���<�;���dR<�6<�C���Y��><��͒���l�'3����w�
Ļ�H�;R@�<�M!�AK�<n����e�<�[U=c�ú8�<��(</D<T1�;8.Y��~<V�V=�`�<A�@�����(N=k�7<�-�<R):)`	=p��K伸�'��2<0{=�h<�|ݼw�Ƽ��\<2����:�;�@ <wΌ���z=����u�=�+̻������|;�;ON��1;�<��<�ю�Fg
��t<�h�y�/=q�v��M=�YH=e���U6�P���ڼ��=,E-=���<�o��T
�<���ɾ =���=�W�<L0=�'�,�J;���;��<I��:��	~��7K�<��h�ą<��:�S���;V:�` �=��<�����C<��Z=��F�.�=�y�1o	�0خ;3������<��=s�H��<�/���>��0�0� e<Nf���=/��</�=S�;�1=6� <�E��W]=�5�|�<�^�;H�>��:1�����<�+�����üb�<8� �F��<��S<|�\<E(����G��7Ѽ��!��=ʘ����E���	���\6�����<�x(�Ar8=�&���t<���<TN��q�gD�p�=�a��<U����:���?|=U��=m�$;ngż��=����R�<勾0Ϲ�y:=J�=�л=�x�c�`���=�O=��<=���r�;���<[a�d�4=lļ;�r<<��<�{K<n/����˼�KG=k[�<5��:��<�s��c=�OR�@��c��s�<Mν��ǻ:KJ=xC[���C����=g����= L�;&��<m�^<(�;�Iټ��9<FZ�<��1�B�<J���"���?-=�KL<���nc<Ϩ=?-W=&#T=����<����$�?A�<"9��VQ�Q\M��弓��9i)l�����]!����=�����s���@U<�������>_>�l��fm�<�n'����<���<:Hf<Kk���E==�?�<��Ҽ��T����<�(�<�O��y�<��6�<-< =�ɢ�Ɨ����;�H˼�E�;᷹�n�d�)Nv���b����<���	;ͼ��j=�Cl�A6=p-�<��=�r�}�=��w;��f�@�G��<�x����A���̼>�#=;�S=���;%�=<O�=�YM�_��<��<i�1=��Ƽ�{C=���;�؊�lTK="�J��s�<�s�<Q���n�<d��z� =p���m~<���^��ݭ�l�׻�Sn=9��:�<Y��K;RU=Kv� ��<A�W�S&ټ�iK<��;�E̼�����ۖ���<�;7�2]���U=;��<��<qQ=6E;<�|=���9D=����ey��.�<y�����<{�$=�W��fػ����<7��<�J�<V$���"��.�3<���7�g�NeB��!��J�;&�B=��p=X�U=�ĵ�h�;�k-=g";�X�8;!͉����;��<}�̼Vg�o���w���Y��%n=�?�<�<<�C=��<�W����<����9�<�F,<>0��
�$�v�������W,]=w�N�-y�=���6���F�<1����<��U�n,+������;�"�=Bt�j�u=�4���Q=")�=�= �E���X<٦��D�=!V���i(<�<U*�<^��:�<�P�<G^�<�ᠼ���e��;S>��۟���<1�c<(Ȯ;w�<?�'�u��=��a=u�<H2�<�^�<�h��v����h=4Jj=3�����	�b���=�ռ0�������=BB�<�a<�Bl���5��1=�d�< �<��μa�ϼ�C�Jߦ;��?;���������=KH�=�<Y<j�:�7=�U���<� =h��D�^�e�*J½� >������Q�=&�<@��.���6�;1ɠ<���v�=ǫ���K��ot=&I.�Ҍ�Ơ=� �|�<�<�� =a\x=��:w��;�<��Z=4���iO������ќn��mQ=o��<;�Ŭ���=e2�<���ۼ��5�{���QAQ�f�׽#]=�n�<�F�<�(4=ЃO�8��C�<�K��G�<èG<A���_ �(v=k1!���;�H��TGM�>�N���Y��C�0辻��y��M�7=`��������RE����� 	=#6�<�P��w!=�|��@D(�����<b�0�0>�<��M�<���:������<����K�=L�S�N3�<�=����c!<E�e<���Ƚ:�ʪ;�_���i���]=-�<
3�<�A=H\�<�JܼЫ���z����6��"����;K�C����}p=F���=6[����B<e��<i��<�39�����-��vr;��4<&/�%�d���g�ņ���#=�p����=�iH=BI=�,5=S-�;�5C:T�<��"�.��=�3'<j���a�!��p=�͵;� �0<?ڼna+<f�=��
=f&�S�-��=&��<�� ��)��!��<O�:*�;��=�jP=+� =��;=��n�y"μ��<�$ȼ�:U=�(��h�
��c��ߣ�<���<�W�d�����9�O#=@�ػN�= [=��g�� =W�����9���p�{�H�D
�9W�Ȟ�<��лs�3��$f<�·=����#=`k����t��0ϼ()�=���b��;@�����E<�$ȼ^��<i��;�|=�}�:�����պrs�< ��<��q=pUE���P<�Y��7K;�x�h�k��c�=���8G-�#�\���>;�>�b���lռb�ڼS�!<.����eɻ#%=Ũ��#<GD"���{=�.��z�꼈�=�י�雵=�W";w���j<�ԭ���N<�޽#�<̫=�Sb���ɶ�Y�=td�=2���Һ8��=�!t=�=C����-�=7q�:H��}�;2�<�)�ߪ]���<�Yb��=�]���E;�F�<[����=v	^=9�9�F�i�<0de�p �=̈́=�� G���Y<�M�<ДZ�����s"�[�	=F��<�D�<�ቺ����.=K����48�:<<H�o=� t=�=�㴺F =d�;i��,F�K�'=ӣm=F�Y��[V��rݻ�Y�<7�c�z���NZu<��V�����M=il:�߲�<����o�<|��D~h;V�w��g<�f�������<�E.�R2���=҂n=��<+~�=*�@= j=�+=ʧ��u_!�v�p��+�;�V�<��'=E�
����M�<��J�2����E�X<�E�"�：��<� �(�;\a=lF*�'4=��<�趼ᴊ���ܼ���=�C���<��;ݾL=}g�<-���	�#�(È�]!��Ս=^`y<�hO���<{�=��4<��<��l��4��d/��a�;_z=�����U�������<>6!;���<+�J��n�';9�V�E=������=콽��<�*y=�2߽�d��%�l=��=`��%<R ��iH=��=��L���.���s��&=b�4<���<�i%=���t�DP�̜�����<`�~t�>�=�E�;�M!�g���<f���мe*�;2�>g9�=;�μPgؼƶ<~.����w�;?7���Cp=�s��r���,=����̩=������<��T�J.��C=WKZ�k'#��x�=ݼ�����J�.(C=��ܼ+�R<@���[4��:~9=rg�6bY=g�����;;=�*�=I�=;�j}<�� =�� <ƞ<��/�%�V�j��<B����<u��<��p<����\�O�3�V���W=]��:J�x��?W��j��ۼ)�=�=�kF=�I,���"=>i1��wC=��<0E��]�r� �Z�g�]�8��<p�9��' =��.�7�<&�+=�fy�ː����}�T�>�m�v=��,��;�:�=s$t=iR:;��<�툽��7s	=Z�<#��<�ǂ<៥�6�<��7=tF$�V�Nf����p�~=��;
�?<��_�(��Zj�G�.<��^�y�c<H.6=4$<&���tK+�P�ú�v#��_�<��='��<:wc���ܼ���=����+�Z��<u�{=�O�<��<�va;]�<�'�C�<��<Y�$=�Ɠ��p[�ђ�!�`�bv�;kd���,=F�<�U�=��ټ4��V��<����^<�#b�<�+��Cn��w7<�лZ�<�z�=ӻ=�dx�P�J<�q�쯥��={f=��L�C.K=.T��=��<���:����&�-�%Y�ƶ�<4z�=��'�gT$=ɳ=�"� �W9N}�</�4=�o<h�-�!�=�D�<+T	�����J3��s�hbv=���=O��R<��=����eHQ=�=g�Z<=���p��<D1ϻs����P�:�<4��K|��K������Ǫ<R==��鼁��<��P�&<VJ���ѽ���
)$�Y�
���=����}�D�m��<$<t>=�H=`�O������H=W��<�� =)��_��<�X�<�&<;c���a��<� �=h�+����;�	p��=��vI8�z*#���O=s嗽e�=�������=��ͽ�!M=V�=�w �4�I=��`=���;NᑼmΔ��`(��r���k>����p�<U8�<Ҡ/�K0=&�=�?��ɹ�ּ͌�<y�G=7߼ـ�=� �;���8��ͼ[���*��������<�g�<D��ly== &�=r�\��X>K�"�i��DY�=�8L��l{���f����;�/�=^�)<l��<��;�C#�AQ<�%��h}ڼ��<B�����<͜�EM�<��ֻ��	�?�.�ԫڼ�]S<���|.��n�<��'�6Z�:�r~�m�U��n���W=�c�<��޼E[��G	�9��<��RH���H�0=�$��oһ�wٻ��B��\���5�#��=E"I���8=i���E�����=5�U��S`=�h�;(��<�u�<��=Vf�DU���p�<k��W���!=1:<��n;|r�<	�;�ˬ9��-��X��=,�<��=Sx����0;�ܠ<�o<��5<>��_ɑ<�*T=Z��W����6����ڻ`=�,S�(=rW��m³�mI�<*�e<aBR=��<p���Q�� ��0����������y=��Y=��O=�D���5�;ke�<�Z��J�<"&�<N�<>{4=6w=T��;46<1T=WC8��ZK�
QX<"c6<�.�<�b�<.V��2��=	B=��<f�����V�὞�$��g�= ��=�ߙ<�"��q�����4�����<�dw�s^�łT=�PQ��Z=2����<B�=���/=�սr�J���[=;ؾ�tM�=d'=����8k����<N���lM����&P=�6�<�;u=��ǽ_�=�Z����w<qÕ=U]�����q=��4��O=�գ:`멼�~��D�_���lR޼�u��mƼ�3]=r�C;~�f;e��;ev9�ZI�<5�H=��s=d=��@���4=�o=��4<�a=(����2m;�'���=@�
<'S��X�<ʁ����<�H&�0�Լtv�;��	=Ǧ+=e~L����<`���ܑ:�3���J<J�?=j��Z��nl2���νx("=�N	�jܷ�T&=P�M=�A�<H\;����_I;�7��p�)M=��6<Rr�|��=V�=�6=�@'=�>'�����(n^���코FF��t�����e���g�;���=)ƒ<6�������z���؍<���</7�;8���ő
���.E�<�,�xؼ��=��m=�����f�`����pƼs��=�i <��m;eى<f4���ٽL.һ��=0��%&�;c��;eY̼T�<=�}��,�3=�AA���;V�<�S�<5�)���n=�?=*�<N�<�<��9A�<;�T�;�]��yм�s=�k=����\+<�<�=p���\�A<~/=OT= !q9n!�<���D��[]=9,��@$ļ�'���[@��-�=��Ż$@�<�O=�Ƽ��}=���=̪�<e��<��P�r�Y<zjn��>���"�I<�ż�,�<�s���8�<x1�<�D=<)�<$�s;�b���+/=�3D<s=֦<�c��L����e��i0�A})=_���E�6���=WÃ=�� ���1��h�;�漫��<�ü�7�#[�;���<��&:�,D���K���Ų<��&�ia"�检<��<h=`&�<b�(<$P)�E�H=�m���B=�=� 8��y�sS�<�1���k�}񐼀J�|�1�.�:ȼ<;����<~��J<=�/�*1�P-+=9F"��QK<��G<�z�<�>��q	ü����a���h⽸HX�R��=�u���=�!�<!���yQ�<�+�Q���;�O��G��F(��x���<���$<0��<iن:�Q'���¼���ڼO���������N݊��9�<���w�<����>@=*gC=?8=hΦ<+.��e�=C{ʼ��v="Q =���<�N<��%��x�:��ռ�$Q=t��	���&ڵ�Ӕ������5꺼2<�=��=��ӽGʨ<�޽�MA=�B�G�A�M���5�����<�~ּ;yt<�d=9�Y�.�%��~�C�=]|���&�=L ��j���{=��}��|�;h�:�ZcW�D���-ռ��>`��=PĽ�.Ļ��V�M}7=���;�-�<���<���<Wۢ�#�;�67=n4[<��d��;=�6��j�=:��=27����2=��Ҽ�^���y����=�PH<A3?=b$=@�P<�/`=����{p=�� =�@�: 2�<����*߼X���Y�=fN3����<�~޼�**=K�-�(��;3!$=K��I��;k>��|�=<�Hɼ��=>	�� �M;���<��H<)=$٬�o^�2S7=w6����<��;1T�J���-�<cNi�+V�86�(�Z<�Oq=�L";�,ؼ:�<k�a���C=���<,�=�	ٽ��^;<�O=�T'�I�3����=�s�<8OC=�v'=�W�H�����1h<l1+���=<r��d|=p1����i�=t�:)��f-=���<����I���4��;Ī�G�����=N��<X��4�<�����-<���<`�=�k�-D�;��g�OM��y�8��"<'*=^����w��`�<�Qe<�T2���,��=��=8��f���l[;�g�����<�<�Y^ͼ�ⴼ��!<"��x����0<��5׼����h��<<�<7�5;5�˽�	�}�:D9úi�	;�fo= ��:��<?�̹�(�����=�"Y�u��ၻk���h)�;�?�8�8����<�k�<b�
=_J=�<c'�<��K�b)3�^W*���=��=��0�t�i�D���ԟ=5�Z�
�d=W���}r�?T,=A�ջ�G5=DAJ<0	o���4�"+=\�B����<��s~
�J�O=֖i=ts��4z=��<k!,=Sl1�@D��F����<�2?�T�/�e�ں# =^�����k漧�G=	<9#�<������P����!j��D�<a�Լ�l�<�>���?D;��Լ\�X;N�>=�n���껔L�;[�@<�ԋ��1�<�m���wֽK� =w�==e����=3B��؂����;���*���7=��X�{=Wϔ<�D��
vL��/#��;�9x
=*�׼�	=�ݻ^A�<�;7iC�\�=Rb<�_�R���,������;5ֽ�Y��2i=��v<@�u='��64�<�$�=��D=���,�ּ�<R�g9h���e�pj�:��<��6<�*+=˰X�
,޼�8����;50��8�н�'=V뵻MQ$=8ָ8��=+�C�Z�߻�Ɖ�������<��kR=���<�筼�܋��B3�5��DE�=��:�X2�<�4/�}{�:���5�<�6<gm���W���=�C,;��m�����Zs�ϫZ=�K��N���,��~ڼ���և���ļ:!=y~漕)�<୼�!L�yR=�<�=�b2=a"��\�(��N5;A���?�<~ )=��Z=y�F=�ܼʀ��P�~��<��f�I���_����%<ܱ== ;N}{����<yQ�<��=�T���8�;���E�<�S�<t�,�!w��)=����A=e� �~M5=p��^m�;!HW=([�����ĕ�3��< " :Æ�<�_ٻX;$<�D#�E8q=��<1tڼXǼq�۹��~�Ư;��/=��<J�����<
=o�5�����V�7��S��M�:,�G=���<P�<������!HF��T=1v:=�Ĥ�����؛��p�]=5�ջv���I��L!O=m��:mゼ2z<j @�Y�\;2�t���s=^��1��;I@I=-b����=u�7=-e"�o{=����W�=������;�J=2.o=�K?�b�?=ui�5���4��<*��<A����X��<Ͷ4�U���=������-�BP_=��P=����w<�a�W������;Z{� `=2�
=��x�<v�<0Q�c��=�,`�_��b�=ڕ׽!�q����;)9C<اf=](�<o��<�܎<�zg�8�O�?��<��O��Ag��-=��0�VR|=с���[<v�<�u�"�$<\�
�mҮ��\���|�<QG���s==�P���/D�In��D1Ͻ�%=�R;��k��
?�o���I�O�4D��g�k<VN��A=��d�	5<W{��iR�<�U���Z;�\�=%=GR=k?��0�ʽ:l�<��<��%=V�9+�=��};���<��߼�7�ҥؼ�����L=w!=��<L|��>}�����r��9L=�W���2=rLK=i�)�A"=Ü�<�YE�{LۼU�b�;ѻ��h�9����=,����<��W=lx��2�=�=���ʻټ7eT=�KM����	�p<n��%���O�[=��<�ţ:�E���S<1��U3`�101=7<�<%4�;��P=D���n�"��ﻼ$�z��=<��5=�D=�G��׊=B��U6��S=���K�<�"s���4=���<g�=<���<t%=������<F�c<&\���<^;!5�z=E�Ѽ(O;���i��p�Nn'����=7��r�<:�ɼEK�;�������_=�%3<׻��;S��T=�<7�=�A��
�<��=<ݞ�m����\���<�;�j8=!)4<\�<�')�Tżj�5=������;ߖ]=g}��0m=�<=�n�<�}H=��;=8�I��}<���;h5��z�<a4D�&Ix=�T$��[A������*;��/:�t��N�<��q<H���e���:<�%������9���%�;˷�<g)�<_��gw+�Z-�dF����V��<��)=�Q�k�+=�;��]Ѣ;�
S=_�;%`<��򻲟J=��8�E�ϼ�cH�ؑ]=��==sI=��n��q=�3�;I;!jN�ޏ�"ۈ<��*��6���<�q�w����g3=B����p <vzi=����>=G�l���f��� =_������=��߆��ĥ�=5�=�*=îG=Rsü���������0�d<��@e�<3%#<7��;�5�<p��o��=V2=�Ȋ<2�*;��E��-�=�RJ;`6�<^�����<�@��<0Ƽ�2F<���Ё�<�ߟ=m����>=#P�<��&=p�]� =�i<���<_�м�?<�<=3�����������nɛ<�-=����.[<W ��~:��e�;��i�
� ��\p�,����n�ut�s���� ��iR<;:%=rW=��-�<5��:�_��l߸=ֹ�������=�4�C�$>��/=���,�=2%7�w�=Ki/�����l=�ڥ<�/�;�?t� a=��G�鼨��<M�(� �b;>��<����ax�����<@=�֤<�V���ż���<�	<ɚ�#7���+;+�<�3D<��!=�놽�B=9KP<�TĽ��=zd���;�<3���@��M�<�2��0��:|<�k)<���<c��<���=�*;.g>=,G=-�	�'�g�؞�:���qHh;p�<F߼��/�ē�<:�<�=*0�<�Z5��Ǜ���	<����:�=I�<�����[��֝P=a��XS=ڥ��Z���,M��bc=��y��<,�=�A	<��K=�4T�$ļz��=��c=�/�
L=�c<����(c�Olͽ0=λ����n��=i�d�4����=��d:�=�e�����q�l�M�<<.�-=�22=h+=���5���Kd=���y�F�=�*��J=����1��+%=h�hM =꯽ۅ��n�9�S=8v�=ǽe�����z<H)<=�#�=HB���d�XP
�f��<h@�nR<o09���{<D�ڼ���=EU�9�o�%]�<+�ݽ�~=5Z��e�<
�;���U��=f�1���.��p=2tP�Cғ:��D�"T�91�<��<�|�=ռ��½��<�,��M�U=���<�(���U�N�!=���'ӛ;� �0�H<�ay=��<�_��w͛<�{���=�<%��<![ɼ2��7aj<Vw�<�<�n6<e1=�Τ�
���W@�j��<%n�=�u���
;��;��5=����2=0h��@�;��<|�N���ż7!='n=�%�<�������<��<pі=L�s=s!<�`�*<�'�����m*=��V�@�Լ��M=0�^��zH=�0�<�F��ņ:�c;���Эn<o\�<ĳ�E'��un�;{sE���i�7�Ͻ�mu=����Bq��=1���(=�@~�	�=񖌽:%�=`?�J��=���<�����/=�&��G=!��;�aZ=R:ҽs�=6O��mt^= 黮�~����9�I<��=�'�̍�<`�=�"��셽gNN=ҿ���'��n-= ��_��E�=����J���8�j(����<�̼�D�O@=յ@=����_=Fb��y;5���4<-�� �n�8r<ɴ
=VKK<��L��i��n��<�e�<Nw�<�o+<�]1<K���Լo̊<F ����Z~�<�=���䱽cfռ�PG=}�?��'��NX=�*.���%��s��zy�@�;x������=\��ec=��;��8<���<e!�����$/�<�w<A~����R��X�<A��\�=k����<�Ά=�.�<A	~�| ��1 =�� =O/w�̘���e��7=�i=T�<�=���;�j���F=�"����=4�=�"�<|��;�V=�v��v��<	WM<0*�;a�&=�	��a�<���<��<t���=��<�r�[�<G��=���.�m=���<�k�<�֨;���s,����<�w
<�wv�=0=p++� Ǽ[>+<������<7���	z&���/=��{<��<��=��<*_,��݉=so=��7=��=�pὌ��-����td�<������l��;/��mA��Ƶ��z�<G��<�vŻ״<I�=��!�@,S�pFc�����C��:�<g���KǍ<Q���H�rp�=T=$=�U&���4�����`:<��C�t���!�'<.��= Y� eP<����'���=&�=�ռ��<�c�<��������K���=J�<� ������ԑ�=��#�z��=l屮�w<<&L���������x�iK��$�G�����;Ϭ�-L�<�������<=���=����=�#b�6?i<i󡽄��ܜ<��1�=ط����/=1�?��"�=�)ݼ��0<�8=Iod<�/=�Ҽ��<��L;>�V=h�w���~<����-#�1�=�2��O�K��0=Ǟ���e�ק<�7��3�F�);x]}�k�C="�<�N���0;��<]v��f�=V_�<ȁz��<�<�r�<_퓼t����X6<��.=�>5�0�3;�_=^�+��d�<�}-����<>�=3T7�^�K�c�1=;?&��*�v=Yu���-=���<�.һ+&<�#{=��ѽ�H=)�V=id���]<F�H;�T=|���cI=��$��;F���b{=��2�,W<�t+��'�xU���,׻�f>�]l�=�aX��7=���n�<Z��83�:5"=������;�a�e�5�Dڻ���H��c��<^츽/�L���C���ü������<H����J3�m =��0<-*�;)�<�0�<���=i�ܽl�F=WM�<wqS=���1���&�=b��<�J����%��_=��<� ��=S��� �Ц�B�
��Z�QX���<�FW���=�c=��v��%�<3���ܮ��Êq��Y���3���?<Xщ<Y^��t����a=�"=1ǡ��S��B�½���<�P<<C��a�=I1<��X%�s�=�4�:�<�=��&��t+=m�<� ���O�<���+�@=؉K=��;:��;??~��=�����=��=�,=Ŧm��H#<M��<�kE<g�ý�G>=�?"=<���Ҩ���77��2��I
�(��.=��?=�[��X�<�ң��i�<\E��0��<.'��_��x��<tI��f7��l3=�A/<���<8����K����1�G��Lü�q�ѽ���<�<.=�@�<�R(���'=P9����<D1<H�W��7ȼ�M�;��<'�=��x�k��6�=�c<$L=;ɋ��.=0c&��/"�@�Z=�;���pk�$�+=R�(=��=�O<��=�:o�6=XG�<�CI�/���h�+ ��D�x욼�p��� �=�{=Ib�<�h&��������=م����;R�����4w��v�%�6��;���� μog�<�Ӝ=�����ٽO�= �<��,��};����YE;�.=^Q'�5��}?=j/i���=\���m �<$T�9>�
�y�0=�L+�dټ�3+�ktv��a��	������R�<DS�=Qv��i�"��L��͡�=��0�)L=�=�?��vQN�Yܶ��*=_r���$����Խ̓3=��_���λ��<����ؾ�u,ͼ�������;�:1��V<=Ŷq�9�3=��<�i�F�,��M8=�����K�;/Y��c}=��
�W�<u/üV��;�c<z=��2��PM�ڶQ�{�'=�E<�W��1�<��������p=��H=?vp<���ɀ��pټ(���g���=׋��	�;���<���p;['=޵<��<l����8d��Ks�kG������G����Z4��E=�L<p��=�=�4p=��>� �����E=��.�����=�XԻx��<�򑼭���.F׼1��<���LӁ<��;�`�C;��&<v�9���;K�=�������n[�����������=���<d�<BQ<�4���;�0f��Ha�<�a=^5=y�L�V �<���zY.<�7W�������?����<*o�<��;" ~�>P=�dp�!|/;Q ���o��[��*���=B�X�rʈ�gd�<���=�N=`��<���Dk=��<�Tf=�lz<����2��-��<�M������`мZ����瀼i�3��z^�� =��!=�7>;X��<r�<p�I=��l���+��4q�<թm���&�1b=��=T�̽�"ӻ���=}�V���<����BF�_��{�(�� �<V=h<Np�<K��<¦w<�W=?�)<L"��6�<+��<�a	=��]:(t%<�q����?��M�:�N���(�<�NJ=o��<��=��=���;M+���!��u�K��5/=�dc�}I��}��;���<c�<]��;u�<~Լ�]���|�����Y����黱<����h���<4���n�=�����{x�y��k��O&<l�=p۸;B�ܲ�=ov~<d]�<��X=1:V������?<�������vm�,�<�� =:�<�����A�񣪼~N˼��º��.��=���<ȃ�;�<�	��Z ���;��lq%���=�G�<@7�<���:�)=�8�:`֘�vI��J=Y.�=�ͻ�N���'	>�3����;-�<a6�ӹ���o�;,y���V�<{�<��%*h=�U,�55��쓼�)=i�=%D#�a�d=�������k���v	=J�y�Ag���+4=�L�=M��S��Vέ=�f(��ʽ<�$0�<a�<�q�9��gռ>�8<Y������;��B<��Ľ�h�=�=#���ge�=2^�Ժ� 7�L�=��3=2@B�-�ӽ>{s=V��)�o='@k�����qs�;GxW:��׼�H=V��<	��;t��<��<���ܸ����!�(�����0�=�}�=��������8�w�2<�L�������<
h~�8���-�<D�<U��9󕅽o�K<k�������V�t)�;��=��;=P���Y����:=�Ԑ����7�=s�s<�iv=��3�5�ټ�L=�1=hN:�=~=xiY��E=<��;qu+�l!<���d��<_�!=�3ѽ���"|��h�-=�m�<e<��!�K�=��p�=b�T�a=��S�M;�=����\�=�˳�e��cV=�.�;��/�=z,�)��;ea�<y�n��yp`<E9���`�=�%~�H�ٽ%��<��a=�9<�:�O�u�Iw�(r�����<�fb<�m�;��-=��m<��<+�k=�:�;�TD���=9�<hp˼ 7��i�6<��r�3l���"<��=D֯��,Z��1�;&Ҽ���M�ڽ�&=6x<������J=��3����<)۾9��V���=F� ��6=A�=���<I�$����;?�<�AO<B���D��]5>�?��<2z7�W�p<�?<.Ν;]��<�μlҜ���B���s�<�ɹ<�]���k<42=ڍ�<ϥ�cv.��U���;���<�� nֽ#�=���/Y=f7�;X=�<��<U������IC�<:¢=�����]�<X�=�௻��輻���g����zQ=/Ȱ;,\&=��\<�=e���h�<�ԓ�ф�;�ͼW���2ػ
<<��`�H��=?t�;�>����=�ul<'0�<O��=�G=�?��B̖<C#)<�)�<1�;����Ի��׼1==� U=)x���A<��=��L;��;���=�����I�,���)R=|���#�D=�*�M�T�;�F8�w	3��v<<H[<�7�IԠ�l��1O��{I�����;�rA�,�"=���1���=ڼ�Hr<�F=���9�>��M=�?$��AY��-=N���Ʊ� L�=@�V=�g<��U=&�(w=��:��'���U��Π�������<`�����H�<���;d���쥂���7=iT���<'/$��I=��@X�����TT�;��7�}�<f�<h8S=��~��]ǻ)Ǡ=�!��mX�=WW�=4��-� ���<�Z߽�!���;K��=~g޽�rӼ��\�b�l�<M�Z�ѫ<���%h~���ϼ,�7;�k�u��;��C=O2L<<Sｈ��=B�̼y��<����ĩ<4E$<j@`=d3ٻ0޽�jR�e*�~C=��:ٛ��^"<��6 <��Y=,��<�c�=�(���K;�%�E;�;d��������=R0��ϱ��҇=�gQ=ΔY�Un�<����˭����<)=�Ȋ�r=~����&�A=�
<l��8=MQȼ�x�;z�m����;3�f,���T\��U	;[M=2�<�f	��m��ZT�< $9=@�(�T�w�P��;E����X<Z�<Ґ��
U ��ɣ:��</E��|��=;��Uud�J�ػ��<��=a !<�9�D�=*"=��l�kI[=L�<�MU�����~ɒ�v�_=K9���|<H�;=���<W]� \J=�a2�v�H��d޽�`}=���<�C�&J4���	=�����fμm�J���<$��=�pB<�7P��T��;x��<��*=���Ze��N<i �=U��4�
=~��<�,ȼ�^����<#�j��/ƻ��f<@ۡ=�]�<f������<�	���ֻҪ�<���<����*G<g�"=��;�*�;�Ҽlgh�+��<���I-���ļۯ�<Io�{���-=
5��1毼�W<x;��4< ��(��<m1�<-O����� �<5J���;<�3=��=��r��������U����<� =S�=7L�=��V�87Y��3��hS�����=�μ*���Sm�\x��G��Ű˼�
�L�3����<b�ü�Vռ^�<������Լsd��;�{=!������R��A':%�2�-"�<������>����� =�"N=��?�=�V=<C̕�u4�=z�T=����ڝL=q칼WY����4�2_=:Ɉ�\X��</l�8ヽ�?=�=*�#�v���=;��4=���S&��l��0=�7�£�=E�1<�ù���,�\��<	����<�j�<�><�6��vӼ���<bl�c�Z<9_��9�;�v��%=
+x����V���_�<0��{���!R��?e�<������z�j��;�<��k�C��=��=j�sQ&=�ݥ��w�<����=�
�޼?��=8|'�EF��L�=!���iV��e9��H�j�<e�@=��|=Q����^�I]&�~����ؼT���u�=���< �f�:�h�<�S���{=����n�t���7=�Ҵ=~���Y#1�X�	<b���g�����l^=�յ�����b����W�T�ܟ�<ö��-a<�Ո9�Ƽq=���y=�R�<~4=G��;}d;���⼀�����z�&\'�����{c�f�<��;�K2=��3��"A��_%�<l���۽<����G<�><�o2=�:����<���=��<�@n��Ы����׼V#=&~%��Ƚ�i�����#���Ƽ]��;9�?@D<�ż5�=Vɞ��m�s�W=�l���b���=ѵv=�]�ruӼ�!R=Q� a���=]&=gɼE�&� �:�p$=������;�O���$���J��n��[����r%=�I�=U=�J��ן�<̖��<�n=�?=�L��Џ�Nü�8a=���ɍV=O٣;��>�f��V� � 1<[C��6h���\���5�;H�&�d=�L�lv㻐%�;+/�o��p�����?�ELԼ[��<�;6�t��;����D<�;��+�/�<�G[<7��<O(Q�`�ݼ8��������;��id��m;��r���*�r�<<���Θ�;_~=ٿ�<Yl�x⓽yT=��1=��=\=���!�<�$;����(�E`=M���=�&��b =uw:T5=LӇ�z�'�+��60R�������p;LQ
=6`B�cU =���=�(�]�����h�ļL0Ż�hu<�{�����5�V��9=��x����W�=k���f=<���j�������D=��=�ѻ�W����ϻq�=	<s����<c����~<�"�;�D�;��S��u_���=}ݗ�I"��U��Hl��V�5�<Ӈo=W�^�uFC<�)=Yͼ�Dᢼ��=�\���Di��=�{��`½!=�KY=�����><F���xfʼ~,d<H�=My]��Pؽl�.�>�<��?= �~��Ҙ�v��=v����=�_�<e���9w��f�=����<�<��!=�$	<��D���/�b�<�㴼�_�]��7Mg<�-���9�p{
��D�{��N>���xڼH˹�@�<�p\<=n��p�]�0���(;G�S<k��B�p���+/=��1�A���Ȓ�;�1ռ��jS�<A֒��Xx��@���=�QX<�1�d����;R݇<c��;Z���z��rؼi�P=
� �l��<�0��/@��Ƃ���S��)����,�����?<�D�҄޼ղ<�R�<�2�D�F�$�^�`�剣< 4�)�k;e���]�����.���E<����#W��2��ݣ��+�ؽټ��ͻ�׶�~:H��$�<�W4<c��ԑ���?=*�߽���0;�<��޽��c��;H�=펀�&V����(����=љ<�4p���.=��e=�|��ƼK�ɽ�_��4xC<���;����z��<��
�{��<?�.=f�^;&"�<��\��>߽�΀=�l�(��=Bb:6hF=Z:�[�%=��K�7��^����4e={���0m�<B��<V��ѹ��8q<��}��s���:��=��8�0A}=߽��p�fM=.�z���=�=�v�;D�S��M��#���<<��2����.5�<�Y;\Ũ:�4=H&.��vۼM�0=�H����b<F���K��ϙL�6�����;o�=,�_�_7���1<ʩ��1��<
��g����b��|<Z���I����X�{Eu��*��w�����;��.�����q��JǛ�3���)���84ս��=���<OL=횋�p�t<��7�=PѼ�<$��[7=�"���ۇ�in��PNo������s;��3V�;��)��G�;
�@���O<�~+��}:��A�B��<z�<&a����<� =��.��#=�W��u.¼qv<�{;�-�;y�=/%�����b=�#���ܼV��.;�; �����z��o�꼵�=	n��Z�<�4&�Z�*�-L�z�	=�/����\=����0�<�M��E�<������N
+���n;[p�!�ýv4(�";�=b�μZ�<ʼ�]��Ү�=g�2=Y��<d��=�L�<�3{��=O� �2����,���<��A;��	�����em��琼������<b�b;�=�r�<
,<S~?<��5�4)����Z����yH=a�8�.Yl���<��=霉����<�>�9�6��3����<�M-�wXX��6ϽR4�"a���Ѽ�BN���#=�#h���»ʰ�������<��k=-�n���|=$����T��(�Z#�i�R<!��B�"���W��S'=7������=�g<zJּ̓���n>�cm����=h�g:8"�cx��xk��-�s��^����)�ӆĽ�i]��B=;�)��M�<�� =5�Y=����ӎy=:�����H�n��5]=d�>��i�<���<�����<ǥ�;������c� W�=3R=<�'<���<�`��	�<T����3ѻ~���X ��%�f���<���;T=20μ���<SL�<?�=�9';;�x���*�O��=J���(S:�2������:-=��-�]�!��R�E�K�)��<�����i�H��S+�mOe=��$�I7 ��GZ���O=�wһ�2�|�ۼ뭊=-(_���Z=����섽�E�;1���$=�ʽ���t����E<Sݥ=9�ǽ���A��_~��M���GU=o��8����\ټ��+<�s������[���~m;�%�?���%c���N]��O�=��9
!���F����jO���<ڇZ���ν��N=��;eo=�1��}����!�=3��=9����<]�<�#+��U��1����Ҩ�<���4�R��Ż� ;ʺ{���<�!&��h����D����8��<��*���⼴ X��t-=2���l=�1�<n4�aj=��=��1����=���m8��r�
=$�b���$�1�<tTb���i��͟��d�=��;�=�\���8�<���k�߼8σ��Y(���=7>7|��-b�C��<x��<!�����x����¼ͣ��9=XR�ƃƻ�&<������5��x=��G=��<G�f�:��<v�̽s�ü=��;��a�)$νn��Y�H�~��T׳<�	=�A[����<4��HN�	м5OO�=����&��K������\I�<𣍻T"<�;��)��1��
?<R��tx�<���3:=�H�<H����S����,�,<Ξ�<%���S��]#�;�z<�L-;���\�y�'N�.b���В��{z��P��:��!�κ���<�U༱=|�q%���T��a��;��;[�%=�Ą��}�<���- �ە���������9��;d��l�(=L:��*-��<2=��v;^$1�`_���>d� �<��P<�!����L��I<\���5=f3�<G�����
=�mJ=fA�<D
�=~gw�V�a����F�TI�:eW�7�)=6E:�C����	=�G作mb��<��^<����]4r����=�ӻu]	�C+��H�<�Rc��(�<\�"��c/�"��˛�O�f<�]�<X�D��<1�E�H��N-�P�g<�MA�q����4�;!������a=z���@�����<��Σ�q@ͼ��`��չk<�.��%�:�,<�j1��>�f~�s �u:<Y�!:{�.<��<|l?:kר��Q�=6|����������0�<����楼i|<�C�<$^м�V��3�D�%��%#�e���s=9w��ґ=<<+�;6}�Å�:P��<��<�1�=m�<A���'�S=��E<1)c�wR)=-����߻LU���a�������=��q��i:�u)	�%k���L=���P�0�en�3�d=�ᙼ}�"��-��7��<�T=�%�<q��m�<�н,��qȼ�#-=�RL�V|*=\qi��м���<qI)=�q������7�}=p&=�&�z+ʼ����_�k=����5W�;�H>:��]�纮����<��ּ��<�2{<UK�:�L���-���Zw�������u=�� ��˼��߻ ���<>=ڴ6;3�K���={�<޼<��=��P<�]J�Dɉ<eKH�-���Ǌ$=��=΢l��~�<C����2<۽=
�<�q�<�L�L��<Y��*Â��<��'�<�
�}L0�t��^N
<�M�_���	r�=�2�<s�=6O>���:��ɼf�=�/�<AY$���μ�0v������^i<��I����<J�޻Jd=ʔ��xǼgO���$=e�<�N"=

ʻGh`�����<��:�u��+F\�A���8���Bͺ��-���!c=��<jd=S@���i���Ѽ�`{��d=�:�=e���Bڮ�ƫ��I�=<�_~�ĵ0;h�������8
=d�=��=�߻b{��{Α;�����W_�kҴ�,Q��'�;�!#�x7��n5�;�]�;r�r=���<�5M����	�P=���zQr�g��<X 4�Dȼ�1˼J
��OR<[�����:�sWҹz׼��<�J=O�<�n�<���Ey��t�=,�t�Ե�<�þ=ݪ����#��r=V��9W�ɽc�z�Մ̽��=�Ey�'ؗ���ܺ>[<��=�Q=$�<�ٶx�Y⼼H1C��%p��ּ]�I��o��
L<�>(=s��<H�C�w�:a�ּTY�2�ýӴ��'���۵��>K��~`н�>�<i&��ϋ�@8�<,(�=e��<P��<����_��\o.��u��_���߽�:�<�by=K�����jk��y�$�	����<8���ܤN�x��୻�G[<a���G��<��~��/�8U=#�3<IP<tH�<mm���-����<N7=�{�Ƽ�Uh�������f�F�?�u��&K=�W��j\��n�T�Cؼ����Q˽ֲ#��,x�sP4���:�3?��)�=+ύ<��=��ٽ< ��K�輜g=`h�;l�W=��!��~=�����ü�񐽏�4�����n=u���<�P=���w��r�� eZ��?F��N4���*=N�	=
%�<�y��C*q��r�<�=A=�h�V���Rh<�G�#��<�϶=�X��e\;yD��C2�@�/�f2��W$���<��Q���
<^	ֽb?�#��)Ӽ��!�s����<��<=�ɻ�g�=b2h��X�S0[<��^�܍�%�N���z��$�V�$�� J�?_���<��=�o� ����X�Kzݽ��M=���<'�8;���|sj�٥1=x�= �;���<c8=2]�8"����p<��S=��p=��=�+P��м��彯�=UG������U��Fnػ���<�0<��ڽ}�<H}<|H�<Nx
<�<j����伣j��h��<�e=<o>!��2=�$�<�u=�<~��<Р'��++=�Y�<z^���Zս��<�G�\0:��3=9G��G�B��n"��Oݽ�^^���<�>=��^=�ɼ18='Ҽ�L]=V�;��߽�5���*� �<w�Z���=t�H�QO�A�O�ҷ�>�ｘ <׷b��뤼w����;�n��>�S�o¼��Y�9伔A��6W��K�����ؼ���<O6Ǽ&������D<��V�L�E��ْ��<����%��S��wƆ��� =7�r=V�O=T��<6S7�ep�_��=%����?2<,-=Փ7=�zԼp��&����ӽ��;z��������5��A���;�Q:;^�2����:�[	��V��^g�?,�B�!=�� =���W�<>�!��J<�֩< ��<��!����<*6�|<H�ؼ�\=*��<6����;���8�=��=�f�;����B�ɼɎ�<P1J�L�x��9��=su�����<�`�oo�:{f#=��.=�s���m�1���C|}=�q�H�7�IX@�NB@;�0�<[�F=l��A���c<ʂ�����<�~׽EP�߄������C=D�E�Z�ż�꨽�ac=o[J<4Vo<ʋ�<�}�<$k(�/�`�Q��AȽ�jؽ�Ӽ�b��h���":�8����<�O"�ݡ����c�o�����=k������K��=
q���jA=5TE<��d�p��̻A���<�yǻM
��1���\2<I�	�d��<Z�)�Ҥ��o���tS�<)�~��	W�䟯�f1�o�$=j߿<K�[�6�V���b�����>���Q���ێ�=�a��������x�O樻�_�K�?=B��Rǽ*V>=~�<�v��m�<�$��Q��f�����<�u�������<�aw���<�,<=��н�׽W����<�;�D�������<��=N��,�ϼ�,=�Ƙ���<=����s꽠?�<%��Kֳ:��6=�³�|R���!�T������E��;�F�a������hG�=�A�<�_�dц���H�O\="��=\�<A ˟���>C��^��D����<7�=�W�=CQ��W�Z���;"Q��P�ǽ~i.=`�ӽ����6"��1 ���μ)W�&����zX<����ĳl���s��		�w�S�F��rc1��EQ<��4��!�R�@��뼕�$�P��t�)<PW�=�R�;�v�<�%�<���;3H��>eX�0	����=��=�����6Y=���<�=��=&��<矻�U)=��><�E_<׽5=�s	����-���R�<��J���5�ǲ���L�����=����CV�z&�����T>�ã�PG�3y⼧Q�=dg�<蟇�Ő���(�<���=!u»�]��/�ҽ�X"��f�8u��ɨ><�<��W�
�z<L<�p.=b9v��4C�Rս��<�2��^=��F<S5U=�_j�oM���d���W=��<��l�<��"���=�u�<�t+<�����@�Ţ���g�<D�q�η�<�E�IFB=�yk;�<E�K�����.��)����Ƽ��<�2B�� �h�˻�� �=�F�d�<6���=��#=��.<��ü���T�p<�c��+�?�5ļ	Iɼ�;8���;b��;���^���~<����#-=�d�;� S�S����U����j�*�ʦl�Ӊ���ν�����6=���<�� sB�����k�|<�l=��̽�މ�n@ϽQށ= <���u<�q=�ޓ<v�л/��<_4W��+��c���m�e���� �O�N��&=�AG�X(r=��=�'k<�3�;[pͻ�����:�ݪ�$=	�FH3<=�<���җ˼$f߽�~=����a����R�`�>=�Ҝ<>[r<1�нx���R����=-�N<N��:�C��y<Ѫ=�~м�����h:1�l�5]S<��V:��Ļ�(�=�==�]'=���<6���?=�2��+�`Kȼ�U.����<%M-�)*=k�&�����q�c?��s#<9���X�C=��ػs�uh�;)r�=��p�w�=�;�����ح=ar�<�����������q<�M�Y�;���{e9={�*�4>�:����0~�k��9m�H=�������<����X=V��O�<*l�<�o:=��g<(.|=�D���4��Pd��=Lj��"T�i�N=	b!���}<Pc�<�_���xƼ�P�h�=3]˽8ල'+�U�<���=rxd=_^����D��g=}����J6�4�ݽ�\�<6��=g%��<伞I���>x������/߼ëL���!(�h�5=� �<������3Q*��v��q�T�Ju����M^�<J}q=�ֆ�B����=��;@���@��������5;#�Q=끢�y�m=s�r���G�ڽS��<����Q.<��'��|5=��;���;���v��t��'3�f� ��8�\�û�`=V�;��Z�ݽ�u���D߽ݲq=h䇽��#��w�<8C;��;�&���H!=p���'<���h�e<%9ݼ�Ӽ��><d�?���<����R㺼�����!�����Q����ؼWv=J�μ!�;܃]�����/_=�m��
��|�5=AY�<��7=':�s&�J\�<0ܼ}�<��#=v[ >ɖ޼A.=�9��<Q����6�j�<�r=�<�F����<	����#=a�;2on=�uۼL)�b�^��;$�K���Ҭ<�
=(/x��7������	8:�Bǽ���;�{|�H��<[��{ϣ=ڐ�<��<]L
<�N^=��G�����Ó���-� L	��x�<�>}=�|ۼ��Ѽ�5a=o��;��<�ڌ��6={3O=4]';�ד��;��ϼְ	����y�V�g�߼'e;�L�<^~�<�9��*�"<�)ؼ��"=�|�;4L�<��<����뼥�,=�]�<?1�+����F8��̷�i�6�Iπ<�呼�(���C-=��S��(H���<�i{=���<>;���U�2)�n�`��<f�-!=��}�|cY��1��=����.�<��<����1<��9=��N<��=<4�	�l����cg���׼�e�<X��*;<Ѻ���Խr�λ����H���^^�;vj=�F<Z�<r�� ?�;-`�N7w��F[���r<S��)M�<��|<��=&��l����Ӽ��(�~KF�-�Z<g6��BO�<�l�ڂ>�_�O=U%=z?�Óۼ���<}���.�,��p����=9=���<jr���] ����"»��C�E�sLY�?��"�/=o�9�;��<����1U��fj_���kc6�ɞ%�g3<���c��!�<�̊=Ĵ��Мi<�*��L�n��y<���;��=F/�:s�C�}�ἡD���=M�>��Q�=q^�;p�Q�}/Ӽ�l=\�8<<�<-ԝ��ʘ<�3�`��=��Q=[v�=F&�<Y�	���<i����ƽFR=����I)=��k����=X� =pć=�#P����<n�k��Ǽ�ۛ����Ȥ�9岼
n��x���JX<gr�Y&�5��a�)=>�����;eF�����j����Y�<����R�<����y�K��&=Φ�;�5ּ��{��G��u��Pǻ�\�;Cr[:������Ҽ\R��P�7/~=�]����=D�����x<�Z5�y/�S�`��w���K9=D��<�k���/S�ks1��5X=^<�.��v�g<
g�I�x�*/<W�߻����^��R�<��p����<��#�#�9r��:1=���?�`���"=���U���P[T<@黻*�<0w��Ez$=���<�Z
=oǕ����=z"���mj=Ǉ�<�F==�'<�>5����Xٕ��
�Yl����=p=E�o;F����G�mۼ�x�<�,�;�=���,i�<�`
���ǽ�'����6;O<�0=O���cG�oش<Z�;��P��抽���������:\I�;t�c;ֽn��	�Za�<H醽`份yY��Ǿ:�P�������<��9ȹ�<߼�<�����Ю=�Z�<�l=A����k�;��<iZ�<�N��ʒ����X�;��&�/b+<w2��Ԋ<nB���b���;<����b\��s��8���=��;I��]�C=x�0�&����4<��g��&�z.�������=_�˼��<O'=b��<+={�ټѨ<����<��輢�=�[��(=�*�=�$��M[=�;Z�*�7�(a�U�h<[o=��<)��<�1O��-<o��;�Ƒ�Ⱥ����ͽ�&D=@ӹ<����<��˻�`@�?�<;��<�$:��޼XI<<Z��<��<�����u ���N;�-�o�J?9|FL<qӺ瑉���="��=xHǼ�nv;`�B��ƙ����/��K�l=Ç#��4=T>��~��<��<9��;v�?��\(;|����C�[D�<�ᔽB�κ �e5�<Տ;*5�
�M�����q[=BN=z0���9x��0%<0s|:j���F�
�%�༖� <�=��Ѽ/E'��{���ۼ��F=���<���;�
��ſ�rV9��?W<����u���D<)?�A��;K�K�]s���(��gl��k�'=����9���������=�}�y���ڳ<�ZK�ΰ<�:�~i=#Dڻ{(n�}=�S��S$�����B����ܐ�y�<s4�<��a=w�����
�rh��g��J���?��ɴ=�P=Zi�;+�=�����J=+�Z�v[;�Ԉ@�qO;��1=��<���U0={Y_� �今;_���qI,�����?�<D�=������<n�!�^�;�'=��y��p�Z�d����SΟ=c(h�d����9<�uv��pV<6s�,���4w�<U�,=�Y��hȉ�)@��x݇�����ۮ�A���眼)����&=y�!�	�<]ծ<��f=�c��i`�;[���=��=�A��uI;s==�d=nI�:�h�<�����}߽��;`�-��qg</�V�z�a��*�Jա=ָ)��"B��c�<�4q:��\<n��<Ɂ0=�-*=��;�Ͽ�B1�� +��Z߼ M��l�;�A�<��,=�g=Լ�<	� <��3=-�6���_�ü}~D����<�t��]�V�P���=1F0=������ܺb��<l$=�><h�<2��<2<
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
*(��s<������<9�'�d~���S�<A��Fl��1�<��:2=
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
,StatefulPartitionedCall/mnist/output/Softmax�
EStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@@*
dtype0*��
value��B��@@*���0O��L:�j���L�/t�&�彦6�0�T<�����07���`�H�<@�i���7��qH�mK�=�g��=�A��>���=�Wɼ�.=�B0>�m=��g���
>M��=�U⽌F��Z�^�0ώ�P<��t���T=�|��t�c=RC��r�������B�?ʸ�y'N�J��:�I[>���lrA���.>IMټ#]Ӽj��<\{�Q��<1��= =���꽽�N���X���;<ߕ�s�2��T۽h]�l�!���6�^�J=��� `s<��7��6W�-��%���
9���<�d�=f�;�[�a�b V<�Ҷ=2:�<s��==v�V�/=���=����>���0=˽jKQ<Gg������3�"=WܹTB���<�<.��={�H�c�(:B6e<j�,���<p��< hk=��L=}�׼������I7<�ӼQ�T=�K`�_|G= ��g�<��&=/�ɽ�-=��=��%<��Y=�6\;�Wh����1>���<�����`D��a�;���ʺx;�>r~�<+4�=�Z�<�E =��=1I�=�Z=�%����ͻ��=VRA>��y=n_�<V��=Pߞ=�$>��=|�">�мLT^��N�=���=E���H��>:>��=��=Z��< D����=�[�=o��S�0��g��ҹ�=�{>(��<��=��=Sx<+P��7(�r��=�������<ы�5>��I<5���K�|��=_�=�Ee>�8�=�us�~̎����>)����ĽQ��==׵�=*r�=�q5=q�=� >,��<��J����<Rp;$A3=�=<81����=�M�=F��=\��<V�8����<�m����}=u8�=��Y<���\=�ڐ<�>{=�ڽ!�>gD�<�)������%��=��=�<�c�=XL�<�!�=1��=��>ޥ�=�-<�[�=���j!��^=?�=¿���p<��m=��=�Z�=&�0������=<r=!B>���=�ȧ9�C{�-��=�Ƣ�A�=1�>o��=�<�=^I=�٬�G�=��<E��=d/�<�����<F�T=�>�=��X��3=����)>^��;�D�=��=�=c�5�=D��=��;v&�<u#��@�_<ݟ��_�:}��=�=nĝ=<<�y<:~�;������=5F����H=�=�F�=�%�=�<�0=b��ɨZ���`<�{~=�"<06�=�伹^<U��<k'�a '�d4���㿼5�='EQ=��@=�+߽(1);Z(�=���=�#�=�	?>r����<��M=�[���Fl>�S�<�.=V���6��t��<�!�����XLJ�d��>c��<L���6�;���f=�_���<����/=Y��;�c$=�[���2�<R�>���=dXe�<��z����
�4�<^�(�$�����?�m�w����=�����L'>�*Լ�CX=�T�M��w��<jn^>�=��{>�;��M��=����1T�6�=�l%>�m >J�c=L}>���cG=&s>�3>,�$>:~�+��>�Fa9oq=Ɂ~�8��=�(=>��<���=8��;�a=Y�C�k���o1\<$n�fr4>��=��g��.)=���=QQB�h����}=��#<����8��<���:�;�r���)ې��َ=?x�N@�=c)��&�<�F�<�#-�	�=��<N-�=IE�=S*�<9�����潷EL<#�<�=���C<�v?=6(u��)�<tZ*<������=��d=���=0k�=dc��4s���n>�q<�Iۼ���<�0=��=H���>=�)н~��l�"�<R'�=i��M���cA;O���ҥ�a�e=Ov/���:��S���;���5u>��~�ʱ��rҼ������= Gx�8�սg[;ф��t���h��ؘ8���D�2����)�X���*��Ȧ��y3�c$-��+:n���\}�<q��<:�s���ݷ�=���ۦl��$���0�m�[<G��<<��=ӕ�>*��=���<M\F=&�z�J=�
>TD���L���ܙ���=B�>ti=Y��<sL�sG��c����m;:����㕽������a���O�O����b��|��[g�y/��R
|�ٷ>E���:/����2�x=\�-=����">��>U4��:�
��	=bҀ��N̽*w=1]��vw=[�C=�����m��t��r{����M��[_�0dk=!�>
�ؽ?��{B/>o��<��$P�������<p�缂�}�4�����w����5Y<+�<;`x��X�������:='K.�y"�=⦼�w<��7<�>*<�B��T8�=����,�Y���=4���H=��K<�>����o���<e�;=<lO�W�<bq���=�����φ���T�<��;,�=�dU="0�=�]=��=�z��«<O�<�>;�G۽^��<�Cs=�ډ=o�]<Ϛq���v�^�{�`�L��[7<	���h�=eM�;D
�*�<���=���='�=y�=l��=�+=a��;g =-�<�,�=�p>,��<�<;�I��p&=k�>���<Q�>\�6�Q=s�ҽu[=�='�R9�!������*�>� 亟X{=�;�<���;�Q=��,�;F]7=s傽:ۼ�&ɽ�����=�Ij=�jd=r�<�.ܼS 4=+��=�`μ˓Q��ͽw =���=�(b���
>��	=*�<��������<���>�o=�q�<H�ټ`�y=��=�L���Q=���>��Z>�B�>�
�>�'꼢l�=�|M>?��=���=#@��7�>����/�=+DE��}=�.D�K��=j[�=���<�}�;��%�3���6�Լ��w=?��<�]�/ݢ;(��=ζ=�m��푽�w�=��<=ev<^܎�B	�:+���+���'>(��	�1=�T�=W�o��8U=�]=(]O=p�P��|�=d-��	E�=��-=NU���L@=R��<<<7�%u�=�O)<>޽�^=Zo���C�=ĸN:����g<G���=k'�|:=%�&=7=�I#�5>�p��Z�4U4<k#>P��<�F���=+�j=uY=Z���3#=cĒ<���<�m=g�I<�r=g��=+1�=k>��ʽ��f<�?h<�1>즡;�M��聼,U��T�<�i�A۽٫���=��<W�=1��<����)�<S]<�����j��Ö��$�;�껼
��hX��(��=<�X�&j�JNj<ހ=8�7�P�R�s�<r=�9�Ž��$���;�l>��#<-�>��=��!���=#�>��9)��;^J�����I=�Bc���=~O>u��='B�=�o�=�{N��>��f=�k��#=6˃=|�5=��<��=.#=�^�=�	}��vE=�U�>z����`�S=�p{�����U"g����=�Q=�g<�RK�:�;P����T=�Xüc�����=Y�w=A�=��%=��d<Ԥ�=��ۺ�� �j=b5=���oJD��ս v:=�p�=��i��L��P>ԁ=5�<|=�#��Nk=^�=@o�6�7=T;�=���=G�=ki�����v�ʼ�@�=�}�:�k���=3�]�ʿ�>N
=��c<���=��V>�/{<�k<�{⺅��<�>�>hx��C�.��J������gz���Q��Z��5��<N�*��%=}�{=BeĽD�B��Z=��>H�=��N<����!z�;�3=f�ü�Jk��-=�Sn�3<�<W��= %��DJ2=�	���&����<��><F`�?l>us�=� =��>V������=�LY>>=k�=��Ͻ�W>Y����6K�=�
�=:�>�!a=ր#<
�r=���=u��=�i�=U��<JQ
>�P$>x';��;~^=�͍<�/���$�;�0;=n$*<A�i<���<��=�9�9�|�=�g��V���Y�<,�=P�<��<���=��.�k[>)���#�R��fn=��4=g�=pr >��~=�ڎ�C�"=:Ȧ=��߼u"<������o��b�S�$=<%< �=D%c�0ݻ<�>���E��;��=<��=�15��V= �0><��V�c=��>�)�<�_�ߍ���y˼ww��H�0��<�[�=�DB���=e�=��w*�<!4뽸� �y�=Mi�<�Q�=P�=L⼃j�4X�=�ɤ<hȖ<���=��=z��=ys�U,�=�����=<b޻��K=΋<���"�o{X<�����徼j1�<�z�=_=2� �=~A��A>����<��=�#�����"���]����<�.�<sgҽV����w�5ܔ=��=m�=��s=�O�=�<������=t:��jK>��u<���=*I�=�G��Lv=��O����;�����=m�p=�r�=!1P��x.=)��+��U��~�=�v���ə:�"=�L��,�0� �J�_�ɽ�uy����ױ�=�S�< �Ӽ�U=�9༵��=�J=��=�ZT=�Wy<��=� Y�Wd@3;�0�
��\�����s=a6�<��=3�=����	]�<Y��)t�<,�<�o��&=�r�<x4��AC<�b�=�8=������<@t�9je�<�F�<�9A=BR(>�-f���Ӻ޶r=�P0�G�=���=��=��<C�,=p��=J�=�<��G�=*�I:p�E<z0���k=�P=yOp<PP_�i\b�Km���=D�X=�1�<��=��ݼ��ۼ�H��&.B<�K>��л�=��ȼ:P��R�6�I�V�>~�<��i�O�:��0�zĀ�x:v=�S"=�lz����<gh���r=��<�4��kg�����;�������9��a.���>f&?�4xV=�>��C=5���S��[��<��X:�=������<r��<�	��Oъ=����9��c�l=RBj�')��ǆ=��K�ViH�C<8ּm��8n��1�=�0ȼ��{�stg�$A6=V��a�ѻ���<0�u�[�?=J��P�;=�5)=�ִ<+鱽�m�-!�*_=Z<z
$<��߽m`(<�=	==�`�;���5uL=Iʹ��8�=�5��������;Q�=����
;uY=މ8��Դ<a���L�u=8_��
��<����G�
�:N�/�ٽ*pB�N6:��g��-3�F�=�"��2����<�V��m?n>��ｪJG��V=oɶ�+P
�V�J�|&�ܩ��6��kI�M�ѽ���<I\s�^�����ѼY ��:	��*��q�4���+�
m�<J>$��h;<���<�e��@0�<o��=Nd�������U�	%���P=���F��$U>
u=Ԥ-=U�8=�L�=Nu��}�>�윽�����tk=��9>)���=��C=.8=�K�=W�L;�"λ�*=�n'<��Y=�?��80���[�V��=�= =���/1^=�BE���-��R=-V=B�!�Y��=9��=)'��"=�XݻZZ9=MQ�=��W�c�ؼ�G=���"=�8<=v¨=mY�=5(=�1�<���:�N�<�=�;�
¾
�=߲�;������=3&���=,�����9�MH�G���{t<F~U=@���<��Y�;�v�=�==���=�a�<�)�����<��;!�>�%Q<���}I"=�\�H�=K�$��k*������=���<�‽��s=���e6>�>��1ӽg�=��1���!=E�u=�f{�'ż�>����=�=�����io=�&�<����:�/ݩ=t-��n�>�V<=��)q�<]��=��?=�%�<����eG|�A\�n&���i=)÷�+ƽ6>�x�<��Z����=tlV=��=8��$�K=m��=�?P=����Q��3�\=��=��<8"�=<0�l%;=ۛ�=l�D=H���Q��lx�=?�h�с�=��ayO=���=1��=P�=-�����4�j���D�=s]$�̓�;�V���Dk�=���:=�=�F�=�N%;
�ο��RΑ���{<kxk=��$9�p	���<H�<}��ލ}=�p�=�P<�8�d����o;q�>8�7=V'=����YI=��<mm=��;� �=���<��=E�=`>��/B���>��$=�$�=JW�<TL>���<�d�=a��=pY�=��<z��l;��|<���=T��=�� �EȽ��L=���=��=��=׋2=���<��;>M��r�>g�U;��\�G<A=S�=�+��B�6�}�>�pŽ���=�X%=�7�/#l��x@�5P=<D*��r��]��<�o=�j[=��8=]�=�t<�(���F=�7�=&����ɽ}�=ŗ�=�E���\=�
ٽ���=?1�=��P=>RH<+�'<���=��>�o���}<cv�=)/�=��;^�� ��</W<͚�=��߼_T�ɂa��a=�i�='�q����<$p~=���<"�2=���:B�T<�t�=La�M�<)B�<�h�=u�����<����at<�Q&����=�.,=o9	=����b<./�O҆��r�=7�<��5=������<��/=a�=�t�=�[ڼ �M����<U��8��9��U\��7�=�z=U"8���=f��z��=�B�<��<"aG�0]��Sx�=>�^�'β<�C=;R>)C���:�=��=4<�=���!�;��;r��o�=��<�
{�.R=�2s=H�<�s=}y
=�C�=/�h���m��{�=�F<�T���Φ<c��<އN=�����P#���F�< m�=���T�	��	�=Ϧv<���ȳ�$q�<�R�=:�<���<����@<x�������=e��=��=��=#����=û�<���V:�<�j�=*=���<�-_�ͺ��j66;�6�<��:�s�=+�=�������<�*=���=j�����c>QL�=/�L=ߺȽ7�~=.=�K�;�T.;��I��W�>������	�l�=�9����k�*:&>F(t;;?�9v�5�:=s ˼��^<Ӳ=@>Q=-L�=��=��=��0��F*>�M�=�jJ�t��;)Vq��d>q�=i�=8 �	�\��|�{¤���=Ly�=B(�=BIc>&Ӻ�ד=d�;�K+�{��=H��=,H�;%j>`�{>���=IT�<N�>w�>�B>(�����>��=��u�&1���
4�(�S=]��<�Q����m�����M��k��<�Υ<sAD=�|�<��><��;���DK�.&3�	���������븼z���CX��<s��=�Z�<� �<�?G��n��O`���0�<����
5��"��U��<��^�4>=ܗ�:��H�Ś5��韼�D=VȻ�Q=g;��9<q(@�M��<����<���U}�<.�ӫ<�+
��ͥ<E�B=6b*<�� �l3=q��[��zf=���KK���=�(� �f��~�g-�����=�4���&=5-��h7;� 򺱷;�@<���=��<�r/�+َ��ݫ=��"=3�m=�?S=�ᴼ�F2��_>���=������=K�ʽ��}�<��<`�0��;>=����`Vz�)N�<f�&>���<���;qH�=.æ���V:VhL=i�=�ʻ��2=V'+��j=d�(=��#�gS;=/��;�q�<)�	<��:�N<o	�=E�|;��1�W�<�R�:���c.���ݼ7�
=�2< �T�3q�<�>=&�����<#J='���c=��.=��'����<���;���)@�9}f�<R�_<ُ�����9�o3�<���:�i�;�X���g���&;I�=6�:=�K�@Q#=���<.Ӽ�=���܅�������=��Y�#C=u���jØ��`�<ؐH��jټp�ۼᑋ=���T�9�����k��=b�T=J#o=�<�	��sN<v� =���!]<a5���\o=:Gx=���<�^�;>��=w5�<�=�=�}�=*�a=�	�<��]E�;�=S�+>N������Я�=w�=�����@y=ja�<�F=@:@��c�E0�=$�伞/U=�:X��<Sx[��鿽B�L<j�>��= �<��<�Y=�Z�=�ϛ=���=�]<,=��<г��B=��=��3�aMN=��~<��B=��=[�:�Ƚ� �;'$=[�=!Na>�^�E��<7��>*@�<�>/[�<�0>۸a��0>���=]�A=��S>���=���=?~��μ<>E�B���nʚ�Xi<r�>��=<��'��2�=��j='��O�'>�K�=5P�\��Ն�=^%=�B�=y�S>�`軞�>�|7/>��ȼ�>;���>� �-k����L��|�>�6�=���<�\��b�=���=x\���-9�oP>f�=E&�>�{˼�=�&���T���]6=���l>A��=V%�>Y뼣��=<��C?�=�8�=�u�����>���=�1<��o��yS�8�d=�㘽p�b��E���ؼ�L=)|;</�v:�1>i`����w=`�*��݆���I>	~��_¼�P�=U�����3�߽7�����z�U��:ʇﻺD����o���z�r|���h<o]��3!1�n�e=����J�v=�Q��S����C��=[�����2�"���^��s>Ke�<� 2�y+�<o�J�>{�=7�>F�{=���=4>k̑�3W;�T��>�n�����<[8�<q;A>~t�<:�C<��=��>�"�=�=[5=���<}E�<~m�<i)>���:Ҡ�=�V�=�+>������m4�={;^����<����uИ���^<��7�-�.P�=���}�<qX���<L8�;!�<|�ü�A�Gt=5��7ȟ=�'3=��=��b!ܼ&Δ;`�C�)yh��U=��=����L�9��I�Ƽ�o =P�N�ӣ	���t=���=�i=9�<9ϐ=�=��'�<�q߼���=�\=�2��z�����Z;����:&=[�;�H=k���5��<b����=�i���� �=�l�a�N�^OۼOo���%�=���=���=U63<	���]�
>qG̽���8i���̽>���U�<b#�<N��X�<k�*�7�ǽhO��t.=L�=I�Q��j����C�l=\>��R�=��h�W�`=��+)=����V@�<O���pv�������Z>� �<�u�,?>���S�(=�w>4��=�#L�{�ͼW�9>/���q�꽺@�=o��=C�<؅��䜼�=_�5=l�|�u=�X�<��ߗ=osH= �>/c��=��F<95
��K<`��<��V=�;_�<=R�<�"Y=�Dн>��<�t�$��<��p����=E$���G��:ɒ<�Y��p<�}�=�n���=�MG=k`&=;=�9�ټ��<��%=B�C�=�m����=�:,=�����11=�Ru=�I�;	��=���=[?5��,4�Nw!>��_�PϘ����Xx;�M(�c�=e�<y�;=*;>[��=I�<��H<E'�;�j3;D6�=+��<*�W=kYy=��;o:�:)
�=�佇9׻����D��|�(�;!�<t�[=�쩽6#� =�ײ��;0=�[=qڷ=����ռ�'`��A���J<�-I�|,=��>0�0����=�w��6�D�;��j�=�z���������LJ6��<,YQ�(�9o=;<-g=�6V=7G=�K��oWU����=`c׼�C�����=��<֜�^�=�J�:C �<�z=?��8�9�GH���~_��=��<�7t�{+�=�,�=M��<���=� <S��=��:�qQ*=��E=b�u���=HP���z�;�aQ��>"��=fқ�8Q=�A�����<F=_���Vg=��=�|�=i��;�d=-IF=:��=��)>��G��9�Ok{=�v>r����i<��c�a���=>�� =R�_��,X=�\<��]</֙���"��i�<���<�S�=�P=ڢ�<�B�����:lC��{ZH��(�o�=�������z	�OyA=ǜ�=��"=�/��P��ړJ=����p2=�~=��R=�a��&�hkh��<���_�;�jT=���;;И�[��=��̽��ʽ�c=�=Y�4���뼗ֶ=��˼7�<�ż=�>�=ǻ�T�H=���j{�=�g��w\;+�<h�O�WUD=x�ͼ�.�<(;�@�<%��`P���l�=k��-�x=��k<=�<�&�;�����v�=�{=�6�Oj�<ږ@=*�U�
^��Q���'���=�{�=p��;MZ�=7q��R��<VɄ=����Z�<&C =UCH>��n��<��P=�A�7�@=wԼ�.�}��=;��=��]��*b�d�"=Ǎ���}�|�=�T�홫��f�>q=��4����<*�=s������G�"=d��g�=��'q�jT�<s��;�"Ľ5��]����={����^�=e#+�#�:�ǧ=�J>>� ��o���=B4<��M=2�R�Ʌi=!7�=7mk=�<�=D�t�`�0=pV�< 1��V�<W���.7�=�dv=Aׅ��L�<q>�=/q���
�~� <�ή�/���l�z�h��<�{=w�H<l|$=�_.=�=�`=�����'=|߷=��;#\���u�i��<�v��qO<��N=ŉ�<:`=/h��9 �Lb>H�=�♽G� <v�ż�2=��0=s=�&�:s,���=�:�<�+=��߼4���1	�k��=��<�e;���=�e�<m�1����:���᢬���&<@��=���=ƿ��گ<X�7=U
�<��!=�/w=��;}x���d��#�F3<+�P�&�t �D����lw�� ��z=<�;��p��}><�)��GͼsO=j�5��Z�<��<�*�j~=$��<j�Y�\�H<��R������:�L->HQF���;kD�=`�����}�iz=���f�tu=xC��j�B<�H��?7��u=cI��	:&��<�м����[�鎽���<�d�<���bE=v�=&��=���=������e;?,<��>�@=>���1ڽ�R�:�m6�0���Ґ�C#���6�<����Y8"�oP=�z��뽡}�<����FVܽ��Q=�z=�O�W
=<�;5�>����=FkD��>�:"�̼zN�<�(o�z觽,C��vy>�P��iY=>����1�<*�<���y2����=�PI<S��$oF�G�=��<d>+�=�JػY�=H�"��VI�#^Ѽ��<��˽���;�)/�����<�<�fJ����̿�=Oݿ<�خ:��=M��%dO�@z=��� �9=��<S>�xj�x��=/�=s�=�1<��������;G�=d$=�A(�;��=�#�<q��<Xj=�k��e�9׎�<�d�<�C<_��<(j��!�=�xкػ�Җx�Y�=<r]�<@8༒��;�=�8B�
6��`޼�Z�=�'=q�=:�=u ��&���yh;���9���=R��<�e�=�M=�;�=�>#<jò��A�<�gj=m�<�-����<sS=mϴ=��<��0�ޠ�����?-��팽1�`������;�;C��<�i3=�t��7X ��8W=������� FO���<�a`����u2�=Ft�<�ƚ=��:f�=ur�=�X �)�S��ꏽ+X���=o�?��C��`�<�ٛ<�=�t���뎃=���=�DO=�'=s�>-g=��Ľ�տ={}����<���;�KD=��-�Kz�=�>q�=񌀽he=/=��/�ؼ�	=J�=3� e��2<�-��=l�>Q�s:G�=��
=vo�3�+=��u=�`���似��<�A<=4�+��5��c�=y���ӼD&����=�(�=�<;=�7�<�=����I�;�F�<&��=��1>t�� ��s��ut;{��87m�*��<��'=�I�=%�N����=�Fi���=���=0
l��}	<g>V<�R�=���<�`;\���3;��X<eg��XA>�>L1�=��G/:3��9�,w=�	>Xiy<�n���o��Z�;��w=���=jg=/��=�N+=W����/�*�7�����=t��<2 ý����=��Ž��=���<Nf=��<[�ջ͛����T�.9�<.i�<���<��=�P;�^P=�;Zy�:�N >w�=2�Yp��h~F��D�;�h�`�<r+<آ�=�u�=���<1�:�Ѵ<��=� �=0�=�Qe��M��n�<�Is=�QS=:(�=�����.^>Vw�=�&>1e>��+=6��{��=#�;#����>�w=����q��<��=]�ӽ�?>R�R<�=t=݂>9BT<4tx=�Nj>l�һ��潙��=�A�<���<|��=>��|��<�%Ἴ��<�X>��1>5_�=���;��	=Q���Q����9=4,�>��G> +>��#�y��=iG��A�G�6�=�!q�bD0>�s�=>�K>�ב=_I>X����->)�
>8��=�<�>x���^�����=��=%c=�Ż;���=�̦��R=P��=�><����%�=���=�>=��=��2=�w���<W$�B�=899�)��t=��=��F�s�h�K�<z�&���V����U��<t9A=Tr-�����R�5=K�H:�!=���=�r�=��>3�ż���<��
��V�=`�>G�7�j,��X<{+�=2��=�v;�(�i	�=��<�΀=X�k=�19�C�=;DW�=uA���ƕ����=��?=d)=~���*�(=�o�=�
Z=h$B<ƹ�P�:>��'��hO�c���`��J1�0�r�؏=2\�<9E�����
��Ӭм�
=�9����ؙ�<@ɪ=����k7�$�>�<�[?<Rv��כּ��:��/=*	.=ڮ��;?�<�\-= ��<�b3=@7A���d=L�S�N1�d�= �=@5�����<mP�LE�<��=�δ�v��0�=�6��h>�S\=�р=�C�=�;�=ܠ�����<��u���<�Ϻ�lX<e ý+�=���7�;�ס�����V!M�cT�<٘�=|%�<$\i���<2L� ]�=��
�v(:ҿS�[�O=�bx�����ƒ���=j�׼4�\�B������^�s�=J*~=��E=��̽	�d?�y��<Ȅ6�h�7�����y��(�=�.<�K��<[��=@����E<P̽h|9=�Yl<�A��G}4<�K�1�<��ü��=EE��N=vO�=���μ��w<A�S��e�;��<x�p=��=,�v=��s<9�=��<V�Y�Ju�=���<��m;���;CK=�	�=������U#=���<�ĵ=d�S<�m�=�P���ν�"���U����V���p=�Tۼ|S��r��<˃�=����hz��]����=-��]<%å�e9�=#�4�==�J=>��=��^�6;Sc�І�.��"~��k2<��<�<K��=��z=�3>m�{=O\�K���W�=uM������5���Y=���:���a���<���<}��=������=]�9=hg.=��	�YF�=�����\�=��q�5��<�F�=�k=�ヾ =G7 =���=����<RR�=ə8<c��<�,>?y�=��=��(���=���<�J��K)0=�1�}F�=~�<:6���f�1��<�輲�G��&�jD�=7k=���<�_�=��<�Ţ<�J�=B<�U<<β����=��v=%g<�`:�yt�<���=�U�=[�`=i�n<߃�=�dv�Y�t�u��7���|��i�ǻ������'=�;�<N�D���m����=GЅ��G)=_%��H��=/׬=X�=_]��d�<�7�J罜ʙ�F������ͽx+i=�܉���<�"�����=�U8��U�=̔�<�A��6�=��L�`��=�5=BJ����;SmH������R@=S�<@��Ox��TA-)=�vp=�A�<<9=���=��_=�v�=
�(��OP��7=k�>�4���ߞ=&؉�0�<�<��ȼ�@'���<���=����n�=���=⩩�~MA=����!=ε.:Ue�=�\=�%=Br5�R��<�s��;V��m>U�Ž��=��K��u�;/G9��=�U==�6#���4=e
�<����~�=��,����Ʌ��v����>��q=��)<w=���(�G�y��.��#>�0عr��=i쵼�s=[~=��D��Y=@����E�<}�=���=��'��d�xL �������=uE�=O�'=�f��c�:�m�<�&P=��=�c�"*��x�<�w��s�=���(&�<$�ü�K�B6G=�wA=��<�+S=��7=�+<�@�=�P���߽�)»*¨=:�ͽ�:�%=�81���,��߬��]�<���ՌT��X����~<�=�>��xs1=#��=t��5�,�L9��uT�����'{y=
�#�Ğ��S��`I<N8������q�U�"��=���=��=�.<ݗ=�Y�Mɵ=�ܓ�"�W���>�J"�臤�9����cټ�ۻ#�<�8�1��<�eٽ�8��=�#=A���+�>�>�2�=iBT=g�S<�}��e-�>�Tm��߰��=輳���r�Z�V^ ;�:d��F����i=#��D�i��Ż�JT=���D�������Z%�B�=���;�H'=�;k=|1E�q���g��o¼+�J���=�~���S;��N���=��$<ϣ]�隇=��P>�K<+��<�1�=hX���#�'_�>�}<`Uu��(���`�=0�g������i=�tӼ {�;���<Lg��vQ�j�T=��M���p��j�<��>gd��C��=�Ҽ�W�<��G��o�<��<�t>�a<UŁ��&�<l�<N_S�U�L�p˙=�v=�2��K=��q=+���������l�=��;��� S@<���="�T��3y<�@<~@ߺ%�1��XQ=�i:��D0��
��� �W&�=!λ�s&=QB >l�U�K>�D=�h��N&=� =V�r������=�Y<w+O�߾=�2�<���=v���>j�>q&����=���=�i�<��H:���mo�<�>�K��b�=�oI����Z�=�0e<|μԳ����W=��=���������p=L00=؁��R��<���=�$=�'��뷎��ı�Бl=nI�=��=�>Gg�=�� ��)=�0�+4�(rƼ��8=�O=��K=�J�:��[;��`=a�s=�uƻ�x<U�7��f[=ׇl=��y�d�s��1<<z~Q�a�=mr�=����b���2�.�;i�=�gO�gC=(�=j��=Ⱦ�<h�	�z=�X�<�� >f��;�mm���f=�b]=�e���f;�?��ڼ��ؼEt���"=�*�@���h6=�ڨ;��:@�<Ӡ�=�y��w�;O,�=-=���d����=��/>/�=Q�i<B�=��<n�C��6�=~�=a�y<�m�=a~a��.�=Bq��Ti<��<Vu�=����\�=5#7=�ɼ�Y�<g(=҆$=.��L�;#$����=9"�=��
;<J��`r�=?�=\%4<��輈k�=�%�=�P��L3��X��[�=��<"J;�x4 �D7=�gL����=���=l~��nfV=������=[�D(>��=T��j������= ���{�=���s#:�jz��*��H+>��㼤G}<�7�<t��<7ݰ��詾�.�=���=��=��=^x]��C=����;�<�?=A�=^�$=�-N�$��=��{=QRd�03�=��>bT"=r>�_Y������������=�=b
¼��h���κ�e����<@���mԺXW��sZ�=xh̽z,�����<�<�HX>i�4<�C3��.��0����ߵ��0���Y��H+u��� =翼���<Ǒ�h[�=�#c�ӣֽ8䚼=I��=���p /=Ȗ��e�;=�<�<	��uU�<6�d;&s��/�*���=Dy�R9V��{�=/�=��>��9<���I�=��罻g+=�g�>��e��TB=�S=��*>�������e!�������_=	M-� ��V�{<2�H�i2�����I��<f���O|�=4�S�|m6���<��w�!�z>���p������1����w�R��U��#V����ļƦ��_P5�d��;�$��;oh��X�=s�Q�� ��iʍ=<��)��<�N��!�#�K�:�_r=1f���|Ӽ\�E=��Y��P~�M��1�J�aĸ���<�݃;��>C��=�~�=���=��q;_����>,��=/�<�4=�(�=누�
�=����5�M�P��Ƚ9[ �P���]�A�R��V9��1K�'���}-�b���G�o�&�����K�=�$��Ͻ�zY;�B �&�x�\����^��7���"��K���;�u�4���#��ʲ�仚�_�o��1̻��ֽ��۽��T���%���A�N5K=�=FO���<�u���E��<ZBF�|u��Y�<i�b�9��ڝA��[f��{�w;��i%�g�Ž��ս���Ȥ��5ǽ���<'$�=�U�<nH�\�V=�>d �<{n�;Ͷ��{�I�$��=����7<�=���<��<���+�=]A�����=g�ɼߔ���=��=Ղg=�/=j�� ��H��<bü�Q==I=��=B���=�1�=��=�*��釼F��8d�=��t=A�<;���6w=N_�=���>r=7/����v<x��<��=¼\r=*E=(�=�k=���Q^~=W��=o_�����h��=�ݶ=vcl<�T�=�2>���=��	>��=[40=1SR<�Y0>R�<R<z�S=PV>��=�P�<�7	=+!;����A!�=�Z=�=檽y$�<�P;=�]>��=#]�= �Ƽ/��=�vK<t�;�k�;X�ɽc��;5��=H���x,�=�=�ʁ=P�={:%=��r;N8ؼ*;��T>�b�=[��=��<Uq=R�u��!=➺;�I=Oc>5n�=�c8>`VF=	�'>)fE<�L>�@2>`=���=$�����=I��<���;׮�=��={�;Pȼ<��=�B<�=ތ�=yX�=ǆr=���=��]:xf=���<Ƥ��8����;A�<!;�=������=5�彿��Wf=������ռ�q:�aE�ͳ�<�m���P =I<�<bd ��º=� i=��<��^���(>?bI�M���U�F@	>�,����=����������=H�7�;$߼v@!=��=�< >Y�<����'8��\=M[=ɮ�<�ѯ��1<#����g�;eVA�U�>=�k�=�b�gO�<�_̼���<��;�k=�@Q��`s=�&<o�d�#��=%��<�z�;�d�=!y8<]��=�(r��r�=��<z��=4=-=3s6=g�<�o���<�=?�^=^����=�T�=Jq��t��=5 =�t�����=�Pk;i��;����H��ǽ[���8��]�
=E=z/�;O��<K����iＺ�q�Aٽ;�|G�5� =�3�6�g�Ye�n�}�Ŭ�=��~�����w=g�q����6�9�=:�=*��Y���Xw���F�=�1ؽ~5��b`>��ռv)���=�;��ݖ=C�==�X=�.��*w>](W=:]����< �x>�u�=�к=�?�<~4�=Fw!��<�;B;�zq;�<����&>ܑQ=��1��Žc%	��s�����s��=4p<��*>_դ=�댻4^=��5�<ā<啖<���'	>���9��=�R=d��=K!
��W�=��>R1?��j*> Te�R
�<l�Ż=�P=�:,>oY]��UܼF�.�ܓ�=S�����=��=m�c���[��岽=�ώ<{�]�TVż�}=hD�=SG=���8��<%�<�Kc;<��x=���nɽ�#;�G=N��<J5�=<���Ӽ�٠<>��<uMf=�=������:E��=x�6�BJƺ&��=�_�;P8=ٝĽө�=7= &N={�}<q��n�������]��<ڇ�<c��;��:�1=��==��R=ݟ�;/�x=�Ol���G>�lλ�>>\��3�=LN~���a�o]�����E��m =�Q�>X�A��(��3b�8��=aZ̻�w=�<�⸽T�&���< L���ګ=��5>�5=�����=�9;��y=��;����;�'0�����>���%�=%�=z6e=1r*�	[�<�@��f����=�gV>�P�=[o���T�;���2<d�:.H�<�')>��>�	>U/w>G/�<e��(��=X��=���;q��=_�>�3=~G��
�>��l�qD���������-&6��P�a0ӽ�h����;a� �U���Ko�!S��g��[~6�UQ�<�-����9=���A	�|k<�2��]���I)�3��=��K�1�>����ig?��,��#��Â��:=�V1���Q4�g���J�a��y��ԋ>H�=h��=�&�<�k}�4U	�.&�K��Q��łP�>�=�]�x��~N�<߸U������=�����'��L��6z<�;�V��:=��9k��ٽ����ƴ���%�ܧ,=&ə���=$�Ľ�~�;�9���`��a=^��= ���~��<@[<��;G�����D<&I�,�m�=���=(�<��x�.��=ī'��v=��=�m�=8��=𝉽�=�<�S;�F�<�_�=��� �<���<�ǟ�?���'=���Y֦<й����D=�x�q��k���Kk��:�;�<]E�=y'ԼWjL�`fF<�`�<p<�����!R�˘�=��>�����8�>���q�j>��B1�=�V�=M�ü2�-��=^z�>�e����qq=0~=BI�<H`>4�<�Q��m�;�[s=�|>�q*>(�>y��=��=%,'>�"ȼ�m=5�\=��r=oh`��ew<8z��Zb>Rӄ���L=n(���!�<:CA=e}���XY�Ĉ>Ҝ�>�PT>��Ki�=N�����G����=(�T>��>���>��>	�;/`�<�>
��>�Ç>V��<�%�>g���Ƌ<F"��|�=����ע��]< �����;Z!8=��s�ݛ:ݎ�=g�=�h�=s�-��Pk=Xμ�\:�ᏽ/뛽)<_���h�n�}=�k	=ֲW<�;<��->R4���2=�	�;�<�=n_S;W	�L�<���܌=(���2	����<8 �=����yN�L���*Y�=�.�͈�=��o=o��<T�=���=*���.==Z[�pp�<6#�:a4���_!�o�Ѽ�9����="��g�=�I ��E�<"�9=֏�=�{ϼG=�=)]�=
㼎Fd=�c=`j�=!Ӻˮ�:y>���� >'�v��.�����<�3�<)j���4^=�-�;)�ǽ�N
�I�=�t=��|���`=���<~B��P=�=jB=q��q�<� ���dn<�j���c=n!<�Q�=!�Q=���=
�|�cv��<���=���=�B\=�t�=�y;���;I`$�S#<� �;#3�=��>\pK=�*�=�]��}}����Q:�=W�=���<>�=e�=�������=�<!=M�	�㕡<ͶT=����
j���K���L=�Q<R}7>Y������<9��h�[f=�4�=1�!9���<v��<�f�<~%=�+�=�@4<m�Q��G<�~=	�s�=�FV���:>R=�5�=�Խ;�P=5-=���:���=6+�Fܽ΅�<��<�9�>a��_>�|>q�=��]��u�=e�ɽ��`<��<���=�u�<Z-ؼi��c=���=cց=#,���1���ܼ$W>�<�<��s>Df����'<k���6���^�<򿁼�e��r��=n�D>�"8�T��R/`=�Y<K/P��U=b��=a�,3��!ѼԆ���}�=H P>�-�=���&��=ޕ�<��ߺ��x�D��Hc���ٻ�A7�=�����%<��ݽ���:��t� ����π>�S<>���=>�;�(��Sw��E�P<��|=lI�=��A>�s|>�u�>A@w<aR;�$>�A�=��y=nݣ�Xo�>�W4<��N��s�<��>�1<*�¼��@<���=�!=�Fi�����Cν�8���8=V4!=8t�痏�:cU=82���=?D�=H���^��=��3�m�U�$�<���=�R�=���f���q{��9U;�ܔ�'e�_���߼_8=�ĭ=%��<a��<�%�<��^��l=����T{6�-K�<���y�=���?��<T��f�=!�>$Z�a<<fC�#��=�e�<R7�=@t����O�ꭈ�uT=�"2>{=�>q<�
�=G+��ē=�߼���<��H=�ΰ<�`�=̰�����=C�h�*��=x1=��g�ע��Q�=��;�!=9|b�Z�&�G��=���틙�q��?!=�C�={�4=^�Z�U#����J='��<��8=^樼j$~�ȑ[=�1L�n��;�aE=Ԅb;�P�<7Z<
���v�;e��=�nu<�=u�<x%2=ñ�=�Q����y������ڼ:�d���мԲ��j�=5ġ�}���p�<Y������J="�=��;6�o��=�=w�<�B�<����X�:=�6?=���;�g�=W�<]�=��<.�L�@G:=u:�<T��<!|��|�=h��+=w
=�L[=@��=��:��b�ü�p8����=�;�6���M$=�m(�hڇ<��u�t�E=���=bW�=�5-=�d�:�z��=�k��s0�<?��<��ѼD�	>�N=O�ӽr^���F��a�������`����=ޒ�<l�Ͻ)�I<c�C=%�O�Ҕ	�r���=pU<��<=A��=�@v=��9<��=9ى�3��zG�Vk�=�O�;�d=�޻=8�==�SJ�m>�M�=�(;=�-]���=O3=0bc=�Z0��
�	L=\������J!O�C��_5@<S�=fR�</��˅<���=e�<��=���=҂�<>,B�G1��F��<>�U��4>����v��;���h�e����N=^���1�u��䁽^���d�7�lQ��,~��7<Ѽ�!j=W�Ž�O�E�ռ�ͼ<�<c΋=*"�=���y��i=l�!=FE������ �	�	�<����FN�D�Q<��=2�>YĽ9�=�8;N6��������`�j+�<=.�<���=*���� =#z=$dɼ�}=�瞽Y>B�M���#>p��=���<-ލ=!��=�g�=�෼<�=KIi=�̼�샼��<0t*<��Y��e�<@'ԼR�="��<��E<��I=L���$��ג����eth��Z<.X;zB�<ER�=�@^=R�G>��=���>�����m�"�C���,=��=�߽�;
<R��=�_�>z�q=N<
R?��ុ+�h��=��E=�",=|������=��<H��=a���#N�=K�0��K�=<�ռmP�=�V�R1Ҽ�����<A��=6~R=�m�=���=��=��=h����#=��Q=1�U>����:�������7��=��F���r=ۓj>�XL>��>�>��S� >�+y>{�=ﻦ=��=�t�>G+���x<�Sz=X�<o|�G��=L���S�N=���=��=�R��@E�J�G=��|� �;9��
Q<޼:o<z<��N�_>�=d�8=�`=�<�=��?<J��=c|�<�Em<c=��=���<ϭC<g�����=�S���93=���=wE>�D=���=�>O=%�;j燽���<��< ��=j�=��<.=b���#=и >=h=Y���H�&���׽��;:U<mژ=�׾�]<<��=�rc=�F��Uu㼀(=δ�=�f=#�=F���5���N�=K=�!9=&ח�r=��=u2@��"м��=_�<�ֽ�s>�����<AB�=���=���=>��<S����=��t=�MJ=�7��%g�
��$�7=X�=gW7=�?=&Y$���&;�e�=��B=sU=Fr���޷�=c��=W�	=ûn�D��=F�>�V�<�\�=韭:nT����;�<�a=�����ؼj��=o*k=g�b=��g�9�>kv��Eҏ��=��C�<F�=;T=\��<$��ɋ��?ݡ<��=�1f�˘l�55=�C�:�v��L`�f�b=� ��_�=ԡ�wY=a>n�==)������=<��=ml�=Ho�.=�ɼ�B3<|[�;`-m=�L�=��%������=�:D�噙�3�����=�R½p9��S�<��`��q<=r�b=�m�R邼�';��p��<��Hش�R�;�_�lm==�_��}=��";���<pd]="}���T�r��P��=Ϡ>�e�=/w8>��=q��w>���=U�M=�[o�ý���<Yl*>i`�-
�<�Fb��;����@��bY�}�&<i����tS=���=m��<����i���t�H���</�=!s|�bP=�(%�P��<�W�<���=7e=���=�=z�=��=���i����=b�k>,?�=���N�����=Z���"]�z���}(=�]�=�<>tә=/@�=XZĻ��6==��=���=J�>���:�0.=RJJ=�S=���<��=�n�����ڷ=���a=B=��>yï��"�:�������ȗM=r5H�V�V=���=��f� [ �v)����=P�༉@=Gp�<��;��$=Tw�<~�v�=��M�g��<��g� >0E�=�d<�c��H=T5L<�g���P�L��=ԑd��e�=�2��X��<�ٻ�\(=)4�=�U==V{=<�=��<k�=�t�mة=����h<6u�;���=Vi�<�Q��.ܭ<#���k<���;�w=�in<גA=�=�Ϙ�)����<�Ʉ=�Z��sɎ=n;�� =��=w�#��)=��s=3�<T��=-�=p2n��o	<r��=I�;�b輡����a7�xw=7݉<�q��� ^<��<it�=C�q=@"�<� �=V3����-��PY=OP=���=��[���)=�����r =��=~�;ϗ׽����.q�=b���,����<~�7=�Y��^ã���<�q�=�B=l�Լ���<踒�J~=�ܸ<7��<QM�Wɦ��(=��U�2��<�"���ؽ���=�^��%gܼ�/�;�G$=��D=�\=��=�#���<V`�=��=�����s>��9�(Y�={^q=Ui=)���'�=�����b���e.�*�ӽ(8�=ʈ=SS=O���A=67=bR6�a�2��/=Ȝ]>��=O����<����}n3=��-�����<����t�=���<&�>�>��g�<w�P<�T�R�Y>�I��5	���e=Zٻ!�"��f��=ъ�z�}����W
��*��;����<���"2�{a�<��`��.��,�U=2��=�i�7�`��e>�R𼻽���4=): �^=� =ɸ=QH=+N�D���%=�Z�=Y
\�2~}�[8S;�n�+���^6>�����t�Ec�=��Ss�<�y�NY=36�=4pl���<���u�<x；�׼�3�=ޜ@=�-�<5��<�9�<cM�<Ǌ��B&@�/��n�<k+�;��g<!ʺ��'�$<ʭ$�f������j����.����*�!&��!��>A>�]�7���=e�=:m��U6��?D�;pD=?�h�K�uH'=�/|=��=�K��#.=� =8��?B�Sm�<s���ʝ����<?�O=�@=�WW���<f��=�~��*�*=MѤ��������<0]���Ƽ���=Cp��V��=
h�<��ս�&3=V�=��/�iG=�R7=�.ʽۙ=侻��0=�.=.�=p�tJ�<&��;@�@�n�U����=D5��l�q#ƽ�H��6��;_��=e�����.�6�&.��A'=�[`�ӈ˽�n���4�,�<�����M��42�<�~�=r��<��9Ē���4�!y�=Ff̼jԺ7��@z<���=��ۼV��B/�<�;=뻗=v�<=���=;���pO��p]�=�}���=��R����>Y.=!�]�b�体⇽�cＯ�f��-���=�
�����*=j��<!|�B#A;D&�=��=Э*�f�|>[<�;�]�<C�X=�h[;6�=��=Ze�=���=�2���ӽ?=���=k�>xۺ=:[�<��<9�=�Ύ�3���Cbc=�Y,>R�= ^�=r�}=;�;��<���=��)<!j�=�ȸ=��P=�ת=Ɩ�=�����z<!�,����*�&���爯=8�7=x��=� ����=��r=�)�;���=�[���<(�ƽ-w�=���<d"�=L��[�M�L�E>��E< O=���Rc=��H�cɀ�p����<��,=*�\;���<�ƚ=�r�<��<�6����
�f`[�%����<տD<̺k>D��==1Ｕ�|��.L>��K=c�U��Ņ>��>���� �Dɵ=x�8=�J���E�=���3h���(�����E:=T�h��Iɽ?�<�v3=7ؼxY�=��V��ż�)�>�L#>��=�t=�7D/=t�8<76��<�z���ʠ�������>H���ԨT=[��=Z?=g���;��2K<!g>�/:*��>�\=TZx�(G�|Vv=2�.=��<����6�=�Ϲ>�cD�=*=N�4�x�s�~����"����`�	��詽<�8<�޻K=m��=<O��=j&=[��=.���}�<UӇ��Z��ɡ=H��<�,�=�N�=�B�{���Hۍ=i黒`R�Ȁ�<�y>\Mc>S4<�j<׍��O�<��=��v�y�=�PG>:�_>�Ӂ>LI���I�<�f�=PЗ=�E>@Gk�1�>�[�<��<�E̼�t�_
�;���`��=-*���=�=�[�=�.����?J�=o�;�f=G�V�,���8�7z�CY�=YP*��{u����)j�jX�=�~-���N=^4b=��󱽐ɘ��W�=ZQ=M�P�]ܒ��"{<�_����=1�Ӹ�<�(=Y�T=���=�Л<��l�
M�;3�=�=+���7�w=��=i����\����8�:��ü�p�� ��:e�	>�Z�A͒=	r�<W^�=�mm=^�%�5W$�,R>+���;��>�h���=�o��f�,<j�J�	M�5�=v�<���>@��t���7�ܼ�X��]S�<�h�=�o���B�[_ǽG�W=�;��>C�>~�e�j�'<?׮;�\�������=��������
<e�#�h�=W6U=�G<Mm��p�|���5���E�e�+=r�=>�+S>���<�z=�%��Ӽ�!��U�=�>j�>�P?>��>�h�2�k���=4$>�.>4P��$�>��E��N��Y�t���=K/�����:�!=ؤ���~;��V=՜��6�}� =�֛=�"����A�"g=<�==��V��B$�ʄݻw7��}�!�.\�=�=�=�����@�h=wJ*�e�N�����j��=����SR���h�ڃ.�݄=ۖ<x�=�"=U��=i����bj�ݻZ=,v>��/:<.�����=Ǆ=H�g=�p<j�=�3w<҅�<Ǭ�=�J��x?A�@������MZ=K�p=�}=��<6���F&�c�+=|��{&�X8<|[B:�\%=8+�<��e�z>7<��(<o+�<���=��=��A��?�<���<�D�t�{=uDE=��˻Z��=��u<Ʌ�<��v����=�b����=��ǽ�>�<�ی=�L=�".=Ӓ��iJ=σ==.�=��<���;G�=3�B=B���E#��ge=r��=@x�=N���h�=�8���]�='>C����<�,�=��<��⼅�=�:<�>6�	�3=�ۼ<��\=�[�=Z�żl"��+=�W�=��
�H">��=n0B��A�<�_y�}�=��n�=��Y=��=��[=Z�Cۗ=�Է�{.=�W�=����N^<CXf�#}�=R]6;B;>mƒ=N�żfƴ=r~=ϔ��5�<M��=	��=��������:č�<*��=�����;=�*����<�<�DV�m��=�mT�"(ۼ�>6�+=O�@��!��V��+>�@�_�)*��g�~��7��O�޾2<��sX����4<iѼ�/�=o�"��<�]<�^_=G(�=ʩ�<(�X=��=׌<Í�<�!�=�#=/4� ��<�G��1[<�J�<��m=���}�B�'=-�O�);�c=���<7Iü�1^=$�<x��<S�J���=��̼����5$�KH�3��=3� =��=O�5����<!�
���������;d�<��I���5=���=C��<f!���'����o=�`���s�<��=�G&=h�<Ak=�7�<ap����<I�;[��=/�*=��!>�J@=zj>>��=́����	���i�:��=۹ļ 6a�ؖ�=CL�=�H)��7=鞤=�3=��1��1��Tv<� �9@�'���M���=��=�}�<	���/���.�t`;�Ӕ=�;�=4J���=��%���Y����%����=��b=��C=�t�<O�ӽ=wi<+l?=:+*�Z,2=����x�<�V��LM���]=�A=C5�=���=���=�u�=��<�V�=Jy�<�=�f{=q��=H&5��T�<a�<<�Ճ=)�Q�I=x���\;S�G:Sg�<���y��ڼ�Ȑ�
�=9�������7�/�QH�=�ɻ._�=��$���/=�=v�(=�<D=��ʼuԃ=��T��ȁ<<��=�E�="�{��j�����=Z��V��� �+��ؼ��ļ���<�[�=z3��[\���T=l�ؽQW>�����*=�4����{=��C��K���Ql�j�m;Hj��}�=-ƻu=j����.=˗J�J3�=������#���O���&���<m<җ]=��<�t�]d �L6=�2d�&A�<��[=��~<U�B��=�=�=��`��r�A��:��<n/�&�6��=�'
��;>}[y��XA����=7k����>�<�Li<�\]<�����5=�U=+��]�7=A�F����O=5(a��=	��<	щ;����OjS=���C�V�J�Y<���7����D����/"=�}\=Ϫ�<�d�=�0�;����5��?���6d<:�=3�=��=Go�=�t=R��<��=��#=��=M�;���<�� =H�<���kX<@�n=��P��?<���{���=f�=��Ի�&^=K)ν2�=��T={�=��=�:�<�KU=otr�����gQ�<�� :�i	<�DL=�{=L�9<�K�6i���G<��%>�e;�`a���<S%=K4�<#i=�J=�!�= z�<ը=��&>�ބ��=ʐs�$���2;9�=6��=�������=/e��Z�<�����M���x<�0ּ+r�j�{�1b���VV����<�e�Ȝ�oF��H�t��V�=B�M;��=W�=�˄=����G*H��r�<Elw�xս�ǼL���ӽB@w��K��OP�ð;<,$�<�T�k� =���=W� ���M=�Q,=�1�<P �{�=�3����(<��P;I�=w�&&=��̼��>��;h�R=ܪ��A=�}��(BN�s(�= � =i���M�w<xG@=%Dغ�=���o�=��>tգ<�l�<��=BM�F��="�)�
S�!��<�<p���ʺ�<��1��2=��?=E�ܽ�H����=���<(-/=G���G��=ϐ׼�����O=*�ѻ�Gc<1u��c�<��U=m�9=��<(���:=�u�=w7d�(�	=Ǌ�;W�B=�y��{h@�+��<�x��j�ѯ�<�v�<'��=� �[>eC=�����ͯ=���b��;���=%��=��A�̰�;n7��rb%�����������"=l	X=��==@G�<St^��28=���<�={F�;�����d�=�V=�y=|IY<I��3D�;��i��Bu��E/=��<m�0�d&=nǻPT�=7�@��}�<�h�;iJK=)��<���<ͻ� +�<���<l��Q�=�X�ջ�=uG����<Y�<�z��c?#����k�=�5�;0Ҙ;6 �T7�=���<��g<s7N=T�N=�(�l�=Ӻ�<�Ev�a�<��[=�\�<�׹<�7��71�=
��=+�<�p��jI=�Ύ����=�d��On����<��j=\n�<�7	�ګ�=y�_<"3=f�M�y�<:Cݽ�ً=lِ�Q�k=�)�՜L=� =��=��:��=��=�/�x�<mg���\�>�>��d�܇�<�5<��i=|�B�#Bi�.l=���=�̍�z&��~*�C�e<�K�=#GE=��`�S�J=�AE<����ּI�<��P=�+�<HŅ=O�<�W������@<��Pl-=������=x�r�%}⽊㜽�^���Q�==y�<T�R=1�ս�s���t<�Wj��;��j���W=mz�;y�<�+>�I>�������p>�_�=޵G��>��>F4X�I"!=�\�=|{�=�̶��@<�ռDּW��Մ��cٌ=���=���<�d= ���($�<�xI=ɯA�jCɽ�u�>]K�<�U=HJ=F�'��<]����V;Ғ>_3׽W�L�38-�=�>��v���>Xm�= ���_ =�n�A;�<x��=�C=~Z"�zg	>R�<슰����<��=�Fx:!�>���=4=#*��W!��;t�<�$!=^>A��=���������-��A֠<TV�=G�����#<� <���=����/�9l�=���<�;�fT=��<AU; �=�#��@=|Z�=�?<t��< T�ը=,��=�k=0ӝ=�d�=�v�<��=N�#=n��<�e���D~��f�=(��<�i�<���=��ּU�<s��;�é=�>���<�;"="R�::��=?3�=��>��<=�T���P[;N�@�\�<2��<�Ԥ<r]�S��;���!��;�i�=|.�<vVw;�缉1<�%=�F������-�>���=�O�R�<=�����l<�͟=0fۻ�	=A���~�3=_H=�k=�j\=Q<��<�T��=ѥ���F=8OK��Υ<�j<�K�=)üÒ��ͪ=kb���oN���*<�r�<0/�;qѝ=7�=a�;����#<�X��.��s:Fh<��;�<Ip=�ޒ�:��=Y��<�$ ��p���(�<��V��=[��;�5���Ө:j���#�����U���q��M�<�4޼�n�=Qh���dB�~�<R=b!.=pLF=.�<(�2us��y�;�Ѽ��y;��=0�<<�ܽ�;<��<_}C>C�<��=\��R�<����*쁽�d�<�0r����<D�=��8��B�<�ʼ��p���
�����'	���w=n�<m�d=�5={���]m1�?B�<�:���=����>���=0�=�p�<�a�;��,��V>$" ��m\=i��=��;�⎽:d)���;��j��� <��¼�k�:��-���Y�j� =�S����<T��;�֐����xE��2=T� =�?����;ř�߉� ���黋I=�&���-�< ��_���;;X!�=�D��[��=��;�P</�H�M�-���}��:'�^"w=5�>u�X=��Q<������[<�#<B ݼ��<H��=�ɓ�/=�<��<��<�D|<*�O=Sn!���ź.�-����<,�~=rd��?z������^���+�N��a=h�=�V=Oٺ��Q=㭄��=Ѩ7:��:�&f>�[T=-�AO�<�<;v��<CC��p<d���� �����<O��=��;f �;S�I=��=���<y#���=�ބ�b?�=���<>��=)�=d��"1j�4i~��b��~@<��/<U(=<�=�;��<��ˏ�<�jͼ��M=�#���8j=�얽�h=�i,<|�(��
A��q=�b@�� �=5N�=�n���������<�=��=j% =�==�v=������D=<��<PM �b��ҹ��������?����;�=�r�;��@/���^���c=��-:�
=��-�R�q=��_���Z<�G�<F�q�ދ�=}���P�S���d��<��=�8\=2�E=gy�;��=*�S=5�<h�u=��=���=��=F�����5��H�<e���y=t�=T̸=�,=��j;���`ۏ<�$NN�A�ųy�%�=��E�Aa�=�ѽ2����o<,�<�<Y�hݼ�z%=��=UA,=�=�4=��=�+9=`o�<�/�<.�> ��=;EG=U>2:��ǽ ?��zƻ�Fٽ=�<�b��o�=>0E=����!�l<��=��*>�����y��~���Y!>g.=yt�E�@>�!= Iս뚽��������;�)�L��f�=U�_=봘<���<�Ս���=����,�r�<6��=n�=?>_=��F=����i����<ON�=��&<G6��d�=c�>�4o=�����B=�u�=xБ�?=��)=�ҡ<�ث:-�=囓=׶8��*y=;��=O,���b�z����ɳ=	��<I	�=�>��:��;;{�����N?=��=Dk�=E! ;��'�s=��=T���ɽ�-m������{=�y�$ ~����<����u��;F�<M���M��H.;ɿ�;�G<MRP=6�e=�㝽�3����>� =J�>:f:\�l:-<�q=$l�=������=��<��>�2�D�,=�� �D����~�,��<�U�;qYȼԓ��P��<C�[��Jf=�dN>��=1�B���>��>�U���R_��pR<�n����ʼ��<�
�=/<�zN=�H�<*�;/�!��r:�9T=�gw>m|�=sb=�H����}��<�4�<�@�<9�>IE;>a�3>�U_>ު@�d�=σ=��5>���=���.�>^%�<�K�=��d<<�=4=D��Tl���h0ʼ&m���w_��"y=t�=3�<+dL��؍�2�ӻ��=���wY;�!;��L-�=��=�͆��W���H�<��=�b:=qo�<=(L=��"�1�(=��<LT��֏^�7�x�P�[����6��=����ܷ<]�=��ҽ7z*��&=�}|=�!x�w@�ǒ=�ʟ=��<l×<�}=����7=ϱ�=�.=hf<`�;=)'�=%�>^��=��=���L!�:��;���=�Ŗ<�~(=����q6=����.X=.c����<@Qt=�#���<�=W�㛿=�R6=�(=� >��=&�0=Q� �y���~=Ɇ�<$��=�m-�i.%���=%?����A<4Bɼ������N���=S�:oo�=ڮ�=l�Q�ɻ����?����,��=8<��T���I�0�/�`<c��<�-�<��=)�	�����w�<��+������o;�s�#u�=��7����=�K�Ǵ�<<�$=Ғ��/�=ҔV<Ҵ=������߻U=�_=�~��Q�F��<�;��G�=�c<����?��X4U��5��g>����=��:ZP�=G׻�# �=�8�=,��=���:��</Dy=\a�=�ĉ=�M�=���<���==�<ӎ�=�=�ː=�A=��������3=2�E(�=�Ǒ=�z��)q=�y=*r��-h����	����<�;�<4�<:^�V~=6��=z۝<�D�=�]��.6=�U=)��32=�D����<JS�Xu{�+U<�����%����F=cB�|���?�H=}������k���=<��=Τa�ڶ`<�>�/>���p=��=(�=����=���!�ܬ:��0=�D�=��p��+�<<������V�;>u��<�Q�<���=���=g���5�=�$x<b#T=@7�<���;
����ź?�/�6�=P.u=`e]�=���N@�=Q��F	@�	=f56=�Ĭ��.�<�a>���>t�=�>?MR=���j���� m=��s�wf���*�<�-�B��>��"���0���<��q���~�ڎq=�v=�U���	��;��<�j�<&�=�y<]����Y=(�=e��g��<��S�CA�;ܙ-=w;�=�R��=EZ�=�陻Y'���8T;}������=�P�>:�D>��=��9<ڰ��̓n=D�y-�=�(>dm>�T >t� >R�n=�D=���<�N~<(9λ$GX�@D�>Yꂼ����A�>��=tc�>�U=ݧB��d�:36.=�<�+�='��� ���R��>��z<�b�<D��=~�5=Eq`;�<�=��=��!�S��^.>�ˏ��@���Z>�٩=�BI=��u=[�<��7��*�=>
��RVF��۹�#2��Fh>��}=�r�=z`�=9�S<!ǃ=�Ͻ}��=/�>�@>�lD=1�=�QҼY�>~���=��>��>iќ>v��>��<�(�=�w<>k>�zI>u��=B��>s�1�@�/�.�4=^���=��׽�P��E����.���0��}��=�����Z�=���ϳ@<p�����f/c=�RɽY�h�=)���1�2ǽ�a ��$�/�(��U�ݞ����u�(;�=���I�#�к����սs��=H���[�!L���"�m���1��ʅ�GY����i���%>�ݽz܂������X���=嶠�����fu;A����#<7E>��<�e��X>l`1�/<W�1��� =������x=?�<��F�U����E�<��B>w�\Oû]̡=2[=���<�$��xҶ=��=pY�̴�<3��<w��������<��=�v
�[x;Z֊=�S�;�D;��=o�=�E`=���ZgĹB�м������;�:��;7Q��[$>��"��
�=��=������'�� ��BR==º���0֕�йռ|s�=���=�4�C]�R���LW;��;���<玅=���<�2��=�:=�#=o�=B
�����s#<��H��kڽ� ����=��<�>�zI=�����Ľ��=�=�<�{=��V=MJ�=�.<��>��T����=sIf��w�<�m�=��p<�>�Ұ=?-=p'�<d<>��z:"\=A��햼��=����_�&���=�:>HG�;(�U�>�IU��|>|�6;�E*���=P�pD�H3	>� ��B�;���F?�_%����<d^=r���y|>�ty�"p�=�� >U���*�A�ːi=^�(<5A�<b^�������/!<%w+=n��=2�=��p�u*==��=r]���}�=3_�;��=�]�=Ñ:=�����=_ǽ"�=��<Io=!� <��={Е<ELn�rqS=�5���2�<��=` ���)��R�Ĝo=��=��	=�㒼����5=���?4
�q�<�'x��|=(�=;�d��U�e�<�J�<#(�<�ڊ�=x�)��A0�W2����=��o���<�������;sT���	<�*F==n��
�pk���{�=�?����!�c;|��=	2�iu�=m�̽��<d�˼����<?>���=�No:�T�=h%m����=�#�9�=V=�8�=��j�ǲ>�$V�k�=�FR<"�=�߽#-�=�*���=1fռ��ļSߎ<|�=Y	0�%6�=�,+>g�2=�����8e=�}=��<N	���K<Gȏ<��@=�¶=�嶽�'�;=-"<��=s~0�+`��~��D���b�=���m�;6lt=@�<�W��zqu�|^c=�V��C��}N�=~�!��qO;�!ڼ�[n�j&��<~����8=��*=I�H�7��>��<.�u� ��5��=V9;��;ii�<=��=-�ɽ�=/�69__߽�F=7�=���;X���3�<,����H��2�^���2�W<>�
�<��L>��@=iݕ�u~>^-=Fa<��=]v)�D��;k�E=я>��=�0���T��W��<i�=N@�>�^=|h��&�����^�-�ǐ�=S������=p/�!ȩ=�Ż�tc�q=�=J��<���l���=�{�1� =�51���/�~=��*>mW�=�4�<�K�<y[�<��=�H�<����d=b@�=��Z<�ߛ�T�e<�0�<A��=��[��X�=Pi�=�>Z=��B;�ޥ<m_=Š�ŉʼ��b�����:ғ<�z��$1�;�!�:s<�<w��=�5Z��ǧ�t��C�������mн�s�=.�c=��;����}۽jQ=�νa�y��B��<�=j����ڽ=CU���7>2����߭=���=��[<�Tռ���\\�=� �W�=ngf�_L�<��S>6ѥ��/9='�
=!�X=�1=�ǩ�k��=2D>��ü��1�H�c>7!����=���<�q�,��Q3<�����>0K��ܤ�=��3�e?>=� ���l��=�#3<��g>{��=�X�<���=�iO��ii�Ke=r��:N=��-=�"�;��=l|@>��W�f>�@Q>��6=e�<>e�=�Ͻ~��=�$��S�=������Ž�����<v*��oȼӔ߽I��%�2>�ӽ�~�?e�=5�7 <c`��͖m��9��S��ސ���!����UA����	�GM����al����b� �H,0<k�e��|�<'N��_���I��:iq�=�����O���ڽ��彁u��Ӌ<8�?�	B/��8��bv�%�
=uVq�қ	����9琴=��=@\>YX�<�C����=��( @=���������v��n$=�y�=iܪ;<����|7��gU=�=���<;�!�e���Z��=���<9�=���={�V����=��=��5=��f= _�� �@=Q1=�8Ž���:X��;6>�@<�=%���q�W����U��=,}��7.M��CX=y�^|��]x�As��1 3��
���p�n >�D�����>v9=i=�G�����oX-;�;�R�=��=6 ��j���f�9���=���<,!>��4�t���� �<m�;���B<��6�,��>� �=M\�>G�<.>��d={��<�W>�Z�7��n�>JϤ>x�=�;4=�d>�j�=��>��<�%>�����1>$�>mC �*�j>�M�>��=wW<�uX>�@�P|	>
��:rв=|�伝�$�5f->�R�>vҒ=\A�><(�=C<>�Ɗ�-�>��=��>��>b�=G!=Ě>��<�T=�#��D�Ǆ�>M+�>���>��'<���>3ҩ���>>��ʽ�w�>�Gd�f �=	�<���=�鯽�/!���7��m�Fw�=��N��ҋ��&�=��=e�ǽ&=�T�<6��=ģ=��Y;�4M=�S���M���=���� �=Xa����=���=�v���<�ɪ=0�)���>�>;��< K�����h=�Z�+����Z��D���d�3g>Z��=%D<C�=7�Y�+���U;��<x�=EE^;v�;�%=��z�-N���'�ۀ�=��%�޽�����Ҩ�]l罸]~:��s���N�cS���vh;�j�@��=3b<Ʉݻ_�>���=|In�|�U=�E�=s�<��~1=[O��>��=�Bd�-�}=�SI�8,d��D�F�:<< =�@x=��=R�E���=� =}�<N0b:Q!	�LH`<�tP=����e>L�>@��= ��d<�<(��7�#:��=���?S�;/��<G���V��=�m�;��<�A��㊟���	<[�=�&��E���h���=���>k=��V��>üG��<R�=w�y����%�V�䙥��~_�K�p��+�<�=�|�m)A����<�C�@�<;�9�%u<*�μ�~W=]��<���=����;`�=Q�=4.=�F�<��B=��<�����=Խ4��=�&�n�=%�E�J������<���=��4=�$�=�+�<�����ܼw�ҽ긼Μս�Y=޾�����;�ً=�؇=��2��橼ͪ�0�=j���/���W����1S���z����y<�-�=�'�Ld=듇����=5�� M�=�,O<dyZ=5�~����=Zԡ=*�=�x�<���<�Mg=��_=���=��<[�=�:N>��=!�.> �={N�=?��=Y�=��}=]cq>�Fz�u)���ʳ=�]���7=�@��\ш�Ð"�{>��^��`b>QO)=N`�=�7 ��ύ=ƒ*=تy��n�;oA�oM>50=���=3�=�	�������<	����=b"�;�C�= d=��>�3��>W>����Z�<��<�C=��Ƚ<O`��&����<��+=`c�����5~������p׽�纼܂�b��$3�=���Ci�=М�=?"=���4�;d|�=�r�=��5='|=�8>X�L����;^Y�Jd���?�/'��~Ȣ;���{�<<��=����H�=)��<��=���S�;�e�<#4�=d�/�A��=�)�=>���	)=� >�]:�\�����=�H��.��\i��CU=K=�<}=�������8��<��V���<چ��EcL<) �������	���B�#��<�m==�ic=Q��=�l��O�=�ؘ�����0c	�I�ջ�?ڼ�(�<ԕ��Ԅ<]#Ž�p=5��=�:�<=~���)/��ҝ���X�I<Ǽu�[��=wp�=
"�xdU=�	���@<�g�<b�<`�g=�C;ӠC���� aM�g눽�e}�B%��[;R;��%����	�<`�^=w��~#���X�6N��B���U&<C������<�';�$i�kjS=�՛�S�[�K=�۽υ���oM�E���l=�zN�3� =&��o�ļK��cN&=�0ƺzo�;�}�$��=�n��s'��� =��>��;���=�)�����= I�<m�;�L����;�I��R��Y�55�=��H=n���Q�>�����9=�/2����;��=�9�<����(h�=���=T��6͛������	��9<�x=s]#�f9���*սЉ��Xf��CS����E����B��=���;�S��8Q<�����=�(,����:|��;B
��x�=Ge���Z�=��;���<�߽��;�O�<��==t���&>��@=�L=L�z�M=K�=�u�=��o<��};��*=^��P����=d�S�y�<�vX<���;�k,�!�����:5� j��&D����=E�y��c:=ں�<�3�|�6<�[�58��<S=d?K<���=V�=�����5����<���[�<q�]������P��D��3X����=���X�=�E=���=عT=�6U����<�>=�D�=�[ü�ֆ;m�3=q�H���=Fr��'�=��%=NA=�҂=Q�ýk?=����9����=���$�w��<�p�����<�7�sټ7w0>�X[��*9���y�������(�Ƚ��(<K�|;N&2��@��$�S>��w��=�S> �к�2���=䚂=�tͽ)F�����;��:���=�\�;3�K<�����(������cͻ��=t;%����G��=��=F��+�=ѻ�=���=O"���=���=����c�A=IZ=V_>��\</
�*H�=��<<�h�>Q�E=E�=�F��U.=��8�z�=(��>?�<��?�8�J>'�u=1��=U�>%���I�s���+[�<9��>+�y�Bp=��<W	=��=Kp'=�0_=�R[=�,�>Y�5>�$_�[��=o�ӽo}�=��>{��=NJ�=��0>�V�=���ڃ�=�~�<��>���=���=�7E>��:����=�,Z�B\�=��"�~a=	]�;�q���K<^v��_ļ���<�ҽ��Y���D���t�(<C�Q��<���=�"�<�������`��=60Ѽ47�=.��=s��=~�����[���$������-=h�1;K�ȼI]�<�w=B�=B�:�<��-]L�oW6��yf�ҡ�=��:�}��DZ=��=�/�<QD�=B������m91=���<|F��(���:�?�X�xԪ��T"=I�ɼ|貽�>�<�ս�L��<9= 鬼X��<洽~��=�Լ=m6=��	�;=8���������B���9>���9�,�< �D=ęW=R9���I�=�F��k��;w!�;���<����WՃ�ĭ==��;�`@=ښ���"7�����Í=C��� �0��<��U�o��=i� ��$��L��=2�<B�?��Np= 1�<��<TI�<ݽ~�c��/=�*��s�=�gD�Mx�?���Q�F<����-�q=;�G�A���O@�rt;=y��8�=��<��i=�_ؽy_ʽ�Ǌ��Z=��+��:�=܏��g��\D$=��>
�ͽ�V���;��@=��U=��>&�=�=�ж:�=�J;Φ�=zG*=h
>��>Nb���f��U7�&Yʽ��&=H0/<��*���%<����#��6�]�Vܰ�d
�=_��<|%<��)=	�8�|�R�k�N>�Ծ<z[=��
���B=�!<=�h%�&��=Cek�#}-��;�2c��U��=r+����K�������4���</���(��<�6�f�=��=�X=���=E=87)=��w�O=���<�i��܅<;��;C9�<Y(�=\/���r�=g/3�W�=�W�<�)=�p0��=j�-=��<�Z=+�x���C=��<=��<SFn��Y#=��=�/>�����=�{q=��ü�`����,=���<�����i=;\�;��?=�͆=p)2����<	3�<<؝��2=.�-=�P��$���=�B���t>lpH=vI�=:�<Ss�QQ�=X!Y����<�ɽv�A>�����p�弥PʼU8�P���2=��>�=�NV|��X�=&�=E<dV��ܴ<��=�];[��=�D�<~��]e�=:������=�lE���;��->��<ϫ>�F_��9��O��&��pL^�|+U=�/�<�^���|=.zV���<m.p���}��H���<kT;���I�=x�½����ڏ��_Ľ��8=Ϙ�0�0�_�D=sړ��t;�2�V��(�$�,<�ּ����=���2T+=0k�;����� ��۷�7f��5C^<�;�<-�ս�o=V�����<�;=�=�9:�2�=|0����j=A<�W�=��W�sPM<�}�=�Lv=�'���I=t[��~�)=��;`�
=��,��2�����</�Լ�n_=>�A=6�k>(�0�lc�L��=d�>>!�=k���������`>��G�=1�T�Y��;)�A��� ;���F�-�P�����$��=7��*��K!7=���xY���<���=���=+-�<�"=|�*��˕<b,=�29��P��5��rz��Bl�=���
�L>֑|=_FP���F��=�	w=c�g=��=ξ	>�&�=b�C=��)=ͩ��OJ���0���=s|�?�	�K�`=rU=7O+�d
�<JP��Y���,O
=��s>Wd�=����>\�b=��<w��������s>��Ek�>B^��Jq�f2�������>��'>-F|�F͗�>�m���=��<;���(��h{#�-��<-f�=�������.�ټ�)��fR=
�t�hk�|�=;�<J���$M;=I��=�LǼ�$5���n���=OkR=�.4=�p��}�U����X ��B���`=`b>���<�ʽ�I%��_Z�ex=��;�Z�<B/��zAy�V9۽�o�	 9�a��=�м���<Q=孛�J��;��<�J<0Q�=�3s<�qv�����A^�j�=Jhd�R�=nԒ�6�<<1���������H����<�Iϼ;��<9 =�P�=xc�='�'=r�;=�ǼIC�<������=^��F&�GA�=��;��=x�;߽�=at�=�I�<��+�^<��V�;���9Ӎ=��=͓v��0=Ьn�J?f�'�:=���<W�<��<1�:���+ת;۷<G����db=YQ�<�P��#{,���z<��J=7f�;/�(��<����P�ř���I
�_*J=�z:�x�P<IK!�W:=�LF�e�<�׼.�;�y޽�c��D�g=�Ud=��<��=�tK<Jm�=Hc�� C<�2�">b<-p	=�;�K�;=x��
:�ɘ=˻��(�i��e��<l��<�oK<oK�= z�<~�N��� ;�8���v>=@�|=kN�=�|5��O;YL/�h�Z=����k�=���:y��<�(���i<��<`�l��l�<��(���#<Ԫ�<G=���X=L��<�7�=���=F���⦻<Z 9�āN�B��=Q�;sa<�z�Y=>��rä<Ucռ���z�=�p����<�'�q�6:JG���<�T�tu_���<S�;��:gM^�Iz�=8z-�:Da=���=7�>��>"��=L�7�+m�=�7�<Pct=w���Ғ�<,��=� =�]<���c�ߑ��T=i7�;Ma��J�<4�=_P6��P|=椖�_��=��<�p���*>Hž=��"�#� >c>A>����>=����<�6�=k�=� x�p)���F��1]=�_{����=�˨<�����G=�i�Z��=G�?��7�=�0½��O<����#�;�;�=���}M,�,-�;��=!S���۽��`�g:��\�<��Q�I�">b5m=yO�=B�����<�(5<\��N�=gB$>�4̷<b	�=6,R�PI�!%�����5?=�f��Ū�<A�������ll �~L=P�ދ_<^�>�@ܽ�G�<����V���]V�;���V��J��<��+=���>� �UZt�_`==Z,�e�����>c�gY<�������=�73<�Bq���#>�!$��!)>��P>���li�=��= s�;�7=�g=��>�s�T[��x<�ܼ��>ނ�=�����j�FU�=Nƪ=����/K=|y�>���=�����K}=����c!=-ei<�Y=�B����	�|�6���>�)=O�>f\޽WZ���2��G��=��=<�`>C`P<��q<�&�=��5���=';<=�2W�<SP�=��=��=68>���$9>�q>��ܽ@�=�����m���ܫZ<����%�=�q��1�=�C�=䃓=��:�+K��]9=���Ij��_�<�=;6��u�d=��=_W=V9= )�=-��=Z�<�%!<N�=��M�d7x��R=�Df��B�<-�ѺH��J�=��7<�y<�H�[Q�<TD=6CH=�oj�Xĉ�1����=@���"K�=��>m�սp�<�^;n��<�V�=�̚�����V���㢼H'��:v=�w��bD����< ��=�[ཤ�=fI��xs�=��x����=������=�"
=
0�=#�;>�����!=�'<=Q� >��3�Ӟ8���2=�U�<dp�=�~��<I=��R���<O*�=��Ƚ{��=�Ng>U�R(<���=�(;�x�=�0U=T�񽿜�ao���>�i�>��=��=tڮ�j
�=��:=+e8�k�=�=�g?>�8y=s��=;7:<a��]�޽�,��q4=0X
=p��=C�=���*
>ݶ ��C>^H>��8=,>� >���=+�V�qO*���h��~=�Z�=ŧg�<��=�X@�H�	=I5�=��C=�%½�U�=�9�=��K�V#�mꤼ@=�MO��:o=�T=��ƼO�6=��u�"�=��<�7����p�Y�$3��"���t����=n�?�e=J�E��<��=�,�;Qϫ;���<3�=��h<n܉���ֽ"�=&�9=��;p��=TJ�<��V� �꼶IL��ە��!���M<��<����R��[�"����c�հ��gD��@{<f�|!	>�"�=waV<&�1��ʫ<�=�;G2�=���=���=�?Ͻ�޼�N6�C�=\;��N��="8ٽ�$F=�t�� �=��a=|il��"�=O?x=��=��P<�̈=�/ɽ���<���=���CJ=��<?��=�1<']�=:�Ķe=�f�=	5���1۽F�:w�˼w	�=
��<�y����]��ǂ�?�n=�h��=��E=I<-Z�Ѳ�l?�CTh=����w��"d���Ya<\���)=0�N=�jνk"�=چ��SȢ=u��h�j��7"����;dS����-9A��:ooK�E7<��<Py�<��;��� >���<ϸ�󇸽��R=��}�=̓�=ul�<D���=�	u���T�- 8�h��="������<NI<�z <�6��Fk�bA�9�0�|���A$k=L��=���=g]��z�=(�2=�7�?i���ۻב=�<��=�ν=�r���,��C�ֺ���D_�<��������>+rD�Î�<�u��&Ի�����=?��=�=G��=h7=�m��������;ll��m;�%�F=;��<�/Z�?��=B��;w�v���"<0�`=4��=n�<��k���<4�$=Hm%��~�<kG=���{ea<�i��U��:�B�=8b��I�%<5��=Ɣ<=L�<;���/�<<���:����l<V�"��4=#�j��*Q�tmƻ2y���L|��-=�G=S#(��w'�Mt	="�@��=\<�ټ�\�븹<���<N<��)����d�=�����<�:�:h�T=G��:�h�=7�=
kQ��
�=�J�=��]�T��=�;�=�zv<�����=�~=VY=7ń���>�.n�X#<&<�3u��0�'�/�=�xe�A�o<�V���O�:e��Q�=
~�=��<^cW=L=fs=�iؼ�bL�rE𽷎�V/.=2q�=�ݼpo�<����;�<_��=5�t6J�I�A���"�~웻T�����;}�q�H��=�6T=w��<-��<c]=wdZ=e&�<@PL��QĽD^=�?���I�<V��<����+�=��h�Z
��f�1L=�H��Q:�Z���H�=ػ�<q�m��ѧ�?��=��8��=|j8=�Ӽb�I=#�Ƚ#�н��m���U��O��<]M=^*@�{=s��=σ��\"�WS�<W�׆�a=H�(>�9<w�p��ޔ�'�Z�>���`�;1T�<3��<���=��=�E���y�r[��v��.�<��<�g���&�g�*>,"(�kyo:��=���DF;�*�<��=��=�h�=,MJ���=��<֥ɼ���=��u�� ��`���k��2�;�+*��:��U��lm=R$=�9=#i��v��*�x=����=h6=!&y<-��<u�f���C=�x�=��	�~�x=��(�F���b�=1<�I�0�h�5��<ļ��M=��̣;�8=��ڽ��,=Nt<��:��-�<��==9b6�+o�qs=[M�<�<0=[�	�C�=�eF�6��=�&h</�<f�I:�ڼ�e,=@"�����<]��,ɽ{�=���<��k������0�<��<!��Q����=��=AÜ��3�<:!=o2�<�<:���`=I̟<�ԑ�B��<�v�=�}=;�,�=�Vu<���;f[���`���9�3�E���=Q[�<g�<�f�=��Ƽ,�=k��$��E\=��=��S<���6q��_Vּ��=Јj=�p��M���kk�<�	��l�c��7i��|=�T��wkd����=Oa{;�l������)~��?�<?ߝ=~���1��<�ч�\/;��T���ݐ���q���ǻ#�= G��;�V]<=��=3$�<�]>Hm���&|=<�)=�N;�5��<e%<��=y ��5=�X�=��Q=?b(�[r/�KV)�H��=���<��K�Us#=�F=��V<(|=�+�����;�f4�W�<��R=p> >�7���<���=5l�=����j#=e)ν��<�&7<��;��9���<��
�`Pݻv�;��~<�w <t�׼�;�&����=A��<��<��=�8� N��r���C~� ��=�b�<���G��=�Ű<(�<'�X�F(�=�F�����<dV���[>=�����,<�Ӟ;CO�=�+�<Q�%�ەw=/F��i^=.�f<=���;YCȼ�U�|��R��9	Ƚ=i� =ioѼ6���{4_���:�c�ԇ=�H�<a?4�_�#>�8�<�8=8J���޽�#l<ݗ��M���bH��-�=�Ga���~<TW�=䎇=3p����:z��:h���ԙ=7�����=�ҟ���<v�;�==��;�l����^�="��w =.`��a��3.>k��7���c*=������=Z�e=���< �＆�= ��<����A=4<}:n��ڸ=e�;=(��==�5��2=���=�m�=9X���j=�(�=z�q�%P�=v�"<D��<z��=Z}H=z6�=ghx=38�<��_�e�r=1@��&;ND�;����G���A�.�R�Ȇ�<��Z=�k;`35=�L=��z<���;� �=,��;��=8Ri=��n��E=�6=��;�O&����<*/��["����{�<�o=��=b5�=��;��=���<��o<�E׼%�<�w�=C�=x��=�;S�[�'��=����N^(�`�<�(_=v�����=�5�<��)>�(;=nG����=���<(�=�I8=�샼b�>��<�㎽���<-i�:�ռ*%7�h����(��ὗ`�_`�>ӎ=�|�n(#�����U8�=i���-��<6���f�g�_�B;$=i��<�'��gq�P~��q<Խ�p��>D=�O���}�=5�)� �E���'��jI>aK�=�i�� 3�<��3=.h��{�Z�6(�=eq >�)<�B�P���ؒ�;`7��Le��`;�襼DpX�E��<\|�<Mۭ=&���l*�_G3��=�+�>F�ۼ爌<y�>U�<ȧ;��\<=UL��(j��[����u�>�� �+9���s��A`����=�>鏆��h�y���=�L��.{y=7�Ҽ��= ���y��=&�=�z��a=������&=�Z=$Ik=^�t��8�<����Ҵ=�=�)���c�<x��7q=<�=�Hn�3��=8��=�� >4��=fK�=���=�4ݽ����q�_�������<?�=�܇�*0���h�=��> #�=��_=�1��RD�U�z�ze޽�n>���kG=<�Ǣ<�7d�<2�l���Ƚ��7�9u��k������<gIz=@������<bD�=�_5<���H��������=f��f�;�A�=����нH�Z�՘�K�,=��%w�<g޽�7W<�=%=G�@=�&�<#D>�h��b�=��7=X��<���ҟW=C�����<��~<��V=l�9<ĸ�=��,� �=�N�;ڿ�����A	��3?<�7��rD�.��<L]/;<��7��=iӐ=q�Ž2M= <V=}X� �<c'��E��9<�=#!m�5�$���	���Y<�gK��b0�6���l!	�ѷ<<�<���<��R=�6=�/=�΢<��t')�G���2���8�V<��:卩=pI4��4a���=y�>��$=6���i=�:|�	Y�<u/Z����<L�A��ĩ�}�g�:܋�ɱ<�$�&竽�o�<�&�=Nl=����%�<�~�<i�=�?�3��= �,=��'�,�=�q���b6=Gr�;��)��0�<6f�=C$�=�1�;a<�Z�=,ý�z=A����ʻ�[<��r�d#�<�ܻgx��e��A0<���\��=�5/�d�|��a�=k���P2�4�d=,�=��|=�����=�C�<Ǡ�;a�ý�9��J��",b=M<�]�A;��������v��=�E�=U���0>=�*�=�0�=4�L;���<��	<�˽U͞��~�<�́<�Q���u��L=��`=a����%��Ur=��~:$�=��8��,�<���2�p<f`�=l�R��ȇ<�����E�R�i=�i���f��>�PS�����.;�l���U=�,�<Z!�Q�Ž�>�=�eļa����f=����*?=��t=o���Ƙ�|�����нj'=���,�����=��4�|>C��;Ѳ'>�
�<�/5;��^��Y���_w�gc�=��+;���=[����Ò=+D;�ȷ�g��[<_a�<�T-=�g|=u伋Y὎(����=��<QʽJ\4��.�=�%r����92>ȧ;�r��� ����=�%{<t9�N:j=C��?˽�< ��-���F:��}�=|�ٽꕥ�]qX�K�V��J��DC=������뼆EB���]��5%=����`�6=�0�=5w�:� �&��<�\>=^�����Fj�����=�=ͼ!����RG�P��<�r,=�F=IJ<�"�<Y�<�L=e`�1�=�60<�;=�
���˼��w<:�1;�*=3����<t�=�`����'s�:��ڽ��>��O=�U?�|R��R<2�=�l=o ������7X��s��hj�uǰ=b�6�W#�<�#�=.�&>��9�����_ɼ:HS=�D�舣:13�=[V,<-=�]�����f�������
i#=�K=���;��7=#|>0�6�n��<-R/=R��=��+�#α��pJ=� <���<ٰ=�p��3Z��Y*�r�ٻ�8���7弌1> R������E<�e�=j�\�u���5!����>Q��=��6<Mn�=�=z��=8s����;�p;�+�<��>x�;n���=*=�k;��\��af=���r����=��Z=��;h��*cu=�,V=	� ��=�<G���W�<���|=nL~�ު��U�0�X�;d^=��^�����ф�=.��J�=&k)�F�O�m�I=�"�
���z������=�1=W�S�Y麢���]b�=���D�=�E�<Q�<��I���<�D�<dZP=���ې=�F����n���Y;�U<$�@�W�	=,�l���n��7v�섎��=���Wq�<�8&;��H�+�m��Z�<���[�.����=�lt==�ܽ��� ��v��=���<ε�=$r=��=��1��4�<R��=Tho=J�$=t�=���<N)P��f�=l_=苎=14b>�Y�<z`">or��ʍS=.�<��d�N�ͻ�<a>	v=Vuּ��C>�ռ�k>
��9-�w��8	�!<�k��{XJ>�4g=��"=�č�L�N=3�h<tᎽ�la<�����a>/e�=7Mj����=�Qݼ�'�;�5��RV_�h?�=�*�={�=*�5��˒=����Z!>��Y>ή����=��)����=n��<V2�=�˼�ē=�F�=�Ik����=��`�ǳ�<%�����=:����e=�R�:�;�<6��=��s=F�<�����SF�\
o=,�g8�;���<�j/9ܸ�=և+=;^;��6=@zB�W/Ľ�?���R=���(�����;-�_=s�`&�<���=U���¼�;�ji<,��f¼Y�>�x,���@�0�=k�4=HW����L�< �����<�%�$��<Q����<��p=�����ݐ'=gμ�o��y�<�r�<!�_<u/𽛭�<�㖼��(=��<�҅=ĐD���;�z�=YV,����=�J�<w�<�ml�>�Q=���<z�<��>'p=T3�</U�=�Z��)����=OԽj�=�3�<�t�=��ʼ`�=�_��=�"�<�=��;���"�M�>_�;��[��_��Ǽ�FY����=ER���x@����="�K<�Ũ���%=����椽��׽E���:/������=o{���q=�X=Q�\���e=�&��{Z
>��*=�e��g��<Ym�q����T=|T���YѽJ�;C�o=='���L�=]@v�ə=*=�'m=��<�Vj=�4=פ�=���<{�[��T�=p+y=�ѩ9��=p흼,k)<����P;F�Y���=Tx��?�8�
B=o�7��8*<�5���c�{.<0��=cNv=�T;G�-=��<e_=�%��׀ü~�׽
n��j�Z=L���g��3`�<�tT�=�^%��=K;_dX��;=����;��E�=��\��>[ջ�xQ��?n��s<���<��\��#�=���:I�W<�yϼ�>>�=V��RE=!x�<,��=	z=�)Y�J}Z��m=�z��u½KK)=#j�<H�<g|l=6=y�=GɻEI���hؽ��&=���"���K=î���k>�9�<j�@��=n?�=_�=��Y=,�=4�<�Y�;��O>pP��௽�Q`�����ýH�->p� �I�r�S鄽���O�Ӽ_j=��<�B�++G>&��Gqu=���<�y;3'�=����%���6�=p�<P����+����\=�D�<���=�]��{��=&�<+=�҄j<�_����=��6>;_*�t�N��>��<��!>Zz��2�<B����:	4}�;��>jv�=AL�=R�ʼ�y=}~<�[��[�h=��S=��O>�%�=w���6�=)�ڼ�+9;��6=�D���c=���=Se�;2�뼚�>"�J�È>3�>u��i٩<@h�Ԑ�/��<�ޜ<�>c ˼p�>��j��8=:�Z��Y�
�A�k��<���=��5��ٯ<�#!>�Ԅ=ކ���=ڼ�w�;��=R�6>������3>9v�>��ܼVH$��>�o۽�>���=���Kz �|�e=W�ؼ�.>>���=Z8�=ІǽO�=��(<ſ`���;=���=���>�TL>�@A�팰=P����O���=jܞ<HC/=�{��:��=�N=<�=)[��@q�>�~�><a�=���=2G
EStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp�a
FStatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:  *
dtype0*�`
value�`B�`  *�`|��=)��;ݦ<���EC�1,�<cݽ�T5=���0�x<�3�<���=7|C=D��=)������'=!?|=|��;�%�3S�H#<�?��4͒<#(������<$"s���=	|�7���鼷{��)��CT��Q���'>��==a<ٟ��+8�$X<�{����Rѽ;�Q<aX<;�5��(E�v��>�k"�y�ټx�½=^q<��<��Ž��T/��ԍ���w(���r=n��îɽ6Ӏ�7Gc=`�H����a�>�>M)ν�Oy�d1c��S�Ų�<�V���V=����+^�\J輋Ze�tʼ�0�=hT��Z�=��Ľ���de=?�^�����ͼ�y���C�������IK�4""��>���x����:�y�Q`�;�o�=:�N:= �]�F�#���,��Q%>�L<��6�u��=�.���^���(�Y�_=c�h�Y;%p�=�T��tw��e����(�/|��˧H���<��=�$�o<B�!�����o��<�U��9 F�)�
<}�,=��(=6��=GS�1<����^�ei�=m5���=��轠���۽#�>�P�=��M�ὤ<=���8���WM�>5��Y�<��Ǽ�d�=�� �m�M�,��=��=J�μ��]o�<o���o	B��
n<ę�F��=��=���K�L��(��6Z���M�<~杽w�� �=7�a��ݽ�T�=�j�=5����9������:(��=v6�<k��4�=��.�a�<#yR��q�Z)�;�>k"(���27=�~O��O�;�	��K������x�=ׄi������<�a��op���v#=w�=oV��sr��������`��O��lu���}+�oNE���?�݌><L�_�h�8�C�����23=p��;�.�R������P�=�%̺tt8�ڒ=���<�≽�0.=�Mr��;�=�<�1�=�9<=����=���ѱ�<�e"�_v��H��P"�:P��;����=Ӏ�=T�ĽW��٦�0�=E��=˶���,��Dr��Ժ[=��!��B����@a�:i��=e�o=8�=gh���=	=���<�F�=<�;�˩=�N���=>����#��Z���=�=lٗ;	���M�=c�<�k��{��;�a�=?{=��u�����j�㼃����s��9]�=�򁽿��+q=�};��+�&�$��y?�%����z���Z����ż,ʿ<�g�=<"?=��2t~��s=�.��w����=��8=�檽�"e�'~�=��n=Ļ1=�Ž�q(<�{�� �;��=��F=���<��;4�H<�(�������̽�<��
��'�<�+˼_��=k�v=�$<+=��>I�=\龽��̼�n6�Ʉ�=ǈ����9�Ѻ2j�uI=k�=�!�詼�y�='�����=7�&�!���g�=J�<t%3	�^z28��Ѽ�Gy�x0=��)���H�0��Xi�����-=��<�=��=��;,S`�T.��������<�Z���۽�j;�*xS�y],<u�=rz�/�=�VG�~�<��pZ�<���:)�<�%1�A�<���e<�k>t�5=+Z=�������;��y�l�;��ȓ=��
�tk;���=��p���Z��U����;��ͼ�=�*0=�rӼ�9˼�s=��:�Y&�5�=�F�=�8	=F���&=�ST��=��=�ћ��?<�r���J+=��=�/�v��l�<w�Ǽ_�9�ܦ�~�<d�+�$�&=]������\Q��X�Zӛ�ƞ!=���<�	.<�gQ� �=�@=�x!�bԙ���O=�@�<DFL=��=k�=W��;���^V=ʌ<�z����	ȼ�uD���<8��
H<bͫ<�O�����i�x}v<��-�[��	�<� ��Kܻ���2>��=���`~)���8g�����F;�v�<�wi���=*���n�ؽ}��E.��쟽{)�=`��<�w߽D�-�2~�=Rw����{�=j�=��e�=}� �gQS��q�Q�=ehp<8Q�<���f�M>6�7��޽d�<bս�T��-��j;Ai,��ƽƊ�����A��<_޺������=�ؼ���j�p�-P���żQ���_K��m*�7��͢���G���>�h�</�R<y�r�m�p��q7��u=W��<�/�;H���)�=�
<;�;�׺�I|�2�
>�9��)!=n/��`S�0S=>e�=��W�����fX�U�����Ӽ�؛=�Ce��tݽlԚ������+�r�g��mJ=X3=-=�<J}<@�ϻ�v�<�=�e=�<%�	>׏(�l�<�����R=Qc�<o7�=�[=f�d=�n��g�e����<���<�oƽ�k�c%�<2�k�x��a��xb�8#>-��=��|=o :p��=����=ٙ�C���VF=���<fX<TC��6M�@F=�6�X)d=Y�^�c�ѽ1��#м���;5�< ��8/�=���<i_<��2�U:$i��U����}�����-�k�=[G��s���� �o|(���=?�l��传�'�Z��=H/==�>P$�޻;��% =D��=�<�G��=��Խ*Q]=K�(<�[����r���?��==9�	��D�=i4�*��==Ǽ��=b���Φ=/í=�=%hD����;v����rl=��U��)� b��KԼf�t�|W�Q6ͻ�tн��<���k~�㨼�u�=��Y�\�<��������{'�=��=��>��:�:�<򿥽�N�fT�<�����l�iy�:�1��>��<����/=U)�;���= ͼ���;uЂ�����	J=�l<=�<�=�}�;��7� ���<̝ý�����Fؽ�x�)و��C���$=u�����\<��5�;+@4�/�=1���as=�hk��a��DӺ=p|�=	�bK�=+&>�>�<,#�=Dρ=��=5������?o�=b �x�;�R�>r�� �	���߼L�Ǽ������=�_=t踽%�����H~�=� �=zA�:��<~��<��V�o�pj���V<WM2��2�<���;F�W�s =�R'=�&�l�+���x� �E�b%�7�=qǣ�������� ��>��.�S<Z�<����w�k�<�����q�=��9�IJ=��=��&=��@�߱6���0:�S=�G= Q���"<��p�>C�ɴ=��}��ɬ<ּ,�=A�����<m���Fc�=¬���樽�c��%��A)�;����ꚽ�#��5N<Yǃ��@�<�ȼ�s;�����a�<�<���=؉$��Y<������	���:����۽mt���=F�Ľ}�=���Ӊ�=���=M}G�N��<`)R=7u&<uf��!�Ѽ�
���/=&#�� ��<?�^=�|�=�+�=�G~=^��[���4��6�=�E=j�'=��<փ*=��O<�ǲ=*�<t�C=j&�=k��=�R=�7��7�=R������6��p��j�<��r��3�=�1�f1�=�2(��=Ҷ��۽~s	;��|�3��;�&#>,><�:G�_|Ǽi�=\V𽣙���ҽ;w�=yqH�1 	<|��-n">Pﶼ#���Y�yR=�Y���=|n켇kk=5ʥ�u_����^=��"�eμl������_=Z!�p�<����~��<vn�<9%���	V<��<E(�=2�=rZ"=8���9�<N�a=�7<\��Q9�4ǘ����Ȅ=�mͽd�w=q\���!��]���=}�=�˺=7
=5�½���=�s�#�=s芽f�=��:`*�<R���(��<g�=LI=8Hý��=О�������=�V�=�B�<]� :*Aֽ��!=Fʽζz;��ܽw�Y<p#��!/�Yz��{�=��=<μ=RVT>�]s��aM<��!�觽A>=�?Ƚ��J���|<��ɚm�?�O�Y}���;�<��^<p�h�qA���<1N���<�T
��5�<�I�=H�=}�=�V=6���p�ҼgΎ<�DĽH.��C�<�E��=�Y=~��`�#����=8��N�U��>��U�A�X��;�=����Zy=f"=���0����6�%? =�k�*�r=���=W��:t����M�OK�=�°:�G��$�����p:>�x���~�D�� 
>o�ڼ�̐7��˽])�tL��XA2�<Sļ�(^��-��j$0:L]ý�]��%�`��x��X=�P0=�=�;�\ڽ����h�.=��7��+D����T�x��`<�<����)�=�	�<��8<᾽�7���1=2�=!��H�H>��Խ��|��e��ގ�342�������c��P%�2|���\4�����ˀ;Նڽ�7��*v�Ǜ0=�����}[=4N>΍���Ľ-g�����<�re=�T��q�)��uA�z�=�)�
�Ӻ���e�b= o��0�<m��?�����|����.�tZ����<=���<U&��<CV��1ɽ���֠0=C��Y>H;N<ǌ�=�r<����6=�N��ҽJKM�+�=6��=�J=%=�a8ܽK�<:�A=� ۼ�]���Z<�o;37�=;`��ꂂ;nM=��;���<m�:M5Ҽ�=s�"=���g���aס���ͽ���=qܖ<u�v=4u`=Y�m��'�D��� �)=�Iy����="d;�d=��q=���=�Y��n9�.��<AG=��8=��ֽ��+=K�=p����W���<�|<=r<>��=Ac�; �)=��!=UP�<�Ͳ=��!=�<ף<=��=�+=���<�%��=T#���L=&zT=�0�<�~�<�^��XX�qȶ;�&��U'=f�=h��<L�<���B�	�z�ݎ�=�=zd=Q���<1=.z�={*K��~�6�(<���Q�=�D��:Y
�b����\3�mʀ=HX�<�
o<Vb�;�U%��!�Az8=Sd�<l��u�;��<M�=mA=��_=�r�<�CM�x#C<^hO<3�Z;�)��#�=�3 :����t�5�6���8r�<^����\<64<6�M�4�����<,�=�_<C
�����&�<��;c	����<���t�y<1��=x1�?;"���=�S��V!������=iq�< �1�"ʟ�������Gƶ=&�\�*����}w��	=�Lܼ�=��T<FB9r��=7��=g��=��
=���=�(M��o��[�<��	�Ў�=ܼ=~>��D��<_Sý��%=�ֽ$R�71�d(<5BO���V�Ž��=4ڂ<��>xҘ���k�Q|����u�g|�<���J�-=VA�^Sj�d�=1���3=ɐ�<�>�;Uc�;M�˽+�X=|��=4�ͽ����Ό�[�^=�� �i$=��_���>�/"�o�뻶�G=Y<G��<������\6��}��n��<u�<��=@�{� ��<!��熌��f�<���=X���Ԃ�w�@���ݼS��<�@=�7�<RQC<f"�<�Ep<s�9�G�;���<��m=�F�	��=�k6�v�=��5ؽPӼ��j��>���(�<:�=S��=9���CH=5r8����Ɵ�=3�:��C�<&���xV��
��<�OS=�3<}�c�=�k(">-�W;���<j��=,X�;�=��<�2ｨR�=��4=f|��?=_��<�ǒ=@��<��=Ŷ�=�|I������'=%ս��ʽ'�3<��*=��A�C��uߐ<FyD�
�ѻ\A4<�F�;G@N�_\;=eTϻp���]¸���ֻ��z�y�,=dG =$f=BX�<�i��"����<:�=G�<�8Ҽ������q::�<w��<;�6�Nv�J7����<R䯻�8�=���36)=vw==ѧ=�E�=L�;IR�=�T�5�Խ���=g�G�+���֒�;�=�������܊=��?=ww��t�6���޽t���럕���L�\_���߼{0�j�������|=ۊ�w�<�����F=B������<�t�� �<T�+�<L��:���i���A��hf�������j<#�=Okż� �hf��8r+���:�)"���G��=��*>�w��bཨ39�6�=xĺ���Y{����<�����_����=�R,=	��<m� <�{��˘��.��зV=��򼤴1�#�9�˼��Wl=��k�ӌ�<vQ��P<<��z<e�x=Z��=�WL=��ûp�<��+�N-��6�=ᛝ=d�?��J�<��<?�;������<sq��;{����h=�����H���Ћ�Mb5�������;�\m<l�����⼛�ƻt2�UXu=@G��C��G���ک߻��㼊�`�3�];�~��j���X�/;�=O4s=�ft=mt=7����)`=P���� =����O�<�j����Ѳ,�
��<��½�缽�1����p�+��(
<��=��n=B��=�>��29;n����k=.M=}-=�o6;p������(훽pݼ|!����Ͻ�=n<Ò������, ��cѼ����~�=u��>�w4=�4�n��Xxͽ��z�#���5E=�ƭ�K@�=���=
s�9��A=���86�(�?=3#=�˼�M��ඖ�|�u=L��7�S�=�4�<�޻W=������<�|�=|.�����f�v:�ȕؼ������'=�����d�<�!��j)�DG�=�х���m<2'�=0����9���=��<��셰�Y�<�`��+�D=���5=<H�p=�%���Q<��<�0��=��=/y�<��ʻ3���������v<�kʻ�C=� >.0?������l��BS=7�=ƶ�{:a;� ��������=�D�:��=!�Ｅ~�=�	J=��l<���=�!����l�~W�S��h���d�
;�%R�� ��� ;��; �|<�!%=h@\�Lr=�(����<��z5����\<�D�<=��z��&l�&;� <�=�X�=kW=> �<�NG�l�9E��dۦ��}ּ^��!�=���=��=���< p���o$��N	�-(���廽~	�=s����;;ԗ����\���t=��!=y��<��o=���(���6�="��<�;��1x#��>���t��a�=���=K=��^���;c�<�:?3�㤘�O�=��)=¾E�e�:l(��{�=o��wW=��h=$c �$�=[85:ZIj=>�=R�<�
��1=�JI=�/7=�FS��O��L�=���<7}�:�=���<0�����ļ�$�<�b�
an������F=�$��<�SN<��(�c<F��&��;Ό;�3�<p8t<J@�<����+=�똼-#!=$�c>��z<v�R�tL����˽)>��x���>=\��|�N=��<	�u��p����ۼͶs���ƻ(eM=���;eZ2=O�=ҹT�����= �l��<���*����N�.!�@}�к<��)���P�7���Q9=��)<^�;_�*�y���C:6���w=Jͨ���+����=��;<eO3<�x�=�h:��=�I���cH<t��M=�؎�^&D<
�<�U�;^=�P�=�r(=y�p=,?��ͪ<��<��c*:<�̘=#�h����S��-B;���1>��'a��?�S��ޑ�<�]��W߽��=}Xm<�l����мL�B��~��=�wf=�n=���=e�=I���Y/];�U���nZ�)�<�]��dz"<�}�<ټ=��=y��:��-�0����+���=
_�W��b�5�$��<�ZN�b���c>�)#=�7߻��<H\�<�7Y����;6ϸ�S�꺽����ܿ����/���;�:�>�<]=�Yl=�{��g=�,ܼ밭�_N2=ego���ǽD���	�=j`�<n#���"�\����1q<դ�<�� <?�C;*z�=egX��/1�X3��v���|�~{�<d�,��1=��Q��>��\�<'�잚�*�d��!��^�ޫ�����oqw�����M��<k<$�Bͬ��e���Wֽ����B$����;E~��`	'=�*����h��<c�<��`>�ˣ;B��8w">s���5=��Ƚ3���쭚=�H��t]<)e�T�=��a�I�=��8��Y尽���=��";4�@=����Z���j���#𽼨U=�e��#%6<e�����H�1���C�R9U�˼�D�����L�;yt��[���м������<�m
>J�<��#=.˽M�=�_Y��u�37�kݜ<u��<�v�A=E����W��<H=���M0`��*=M�@�f5��Q3<��K���O�=Z�ƽ�Z̼�t�<FD콧F=�B����<�2ݻ	��#B,�ϖ�=p���J�=c��U���S=W塚Y���[���ݦ=���<7aڼ��Ҽ��=�_49�N�;�<k�h
�w	��B6���O�=�ʼ)$J���>�����<���;�E
�����s<S�����=O������<%����c�ش��!�<��0��A���z˽�n��蘽Ȣ۽��;Mi���t�� ��=���&�</ݬ��]K���>�uƽ�,<f�>!����=
��(�m�_��3}���9��,0<��Z4�����<料��(���?~��S�=NI	������q'�_L=1-(�>	��m�<ݎ���@>7�=�=J۽sKf����;��G=�i?�:���6�<�*<��<8����A<�L=�;�=�F+=O��=���=�䀼F&)���=a���3�F=L�
<�>>)	Ӽ�]�<#�I=ˀ{������;x �=ۍ��;�\����;
%�;"ln= �|=��W<���=b�=R��1���'鮽�0�<����$>3���2����ݼ��4�����5�{�h�,�H��X����0�ͫ�;Ybt�����"��mɽ�AսW���3D���:>ҽ*2�=y ��Žuq��:�:��(��ʧ���½?�ܼQ�=p0�<Z~��G���i�= A���*�N��8�=�љ�cڿ�����	�G��"�<F�,<��/=�0m=�Ᵹ�S=[\=���Į����w�=q�=æ; %�=E�<Nx�2�̼W��ʮ}=��L��&{=�ㅽ�e:^�<N�����j<���;ᙰ��О��@���%�i����.<Y����[��돽�8���Y@�w�<��%�D�-�DzU���5=0~J:𭂽q,�=���<�ݽ�Hw��>J�^v?�Y^t=P�?�9������=
Q����:��ܦ��y�=|���騄<uZ1<����N t<m�мtO�����=�
=�=m��p0��!rU��� �R��UC=��=�RE<L�)�4�<% ���[
�M]���>�z��<��=Ϳ�<��<��˽���n�<�3�<�(	="=Z��=�S=b�=��!�mt�<��L;���=ΆR=�ڜ<z^�P��=��0�z��I��^YE�`�4=���:��> ��<�����X=�c�P�ټM�L<�pX<"Sb=���p���U�W;PQ�=|�=h5�=�b����!����=���Χ=�u1�E�<uR=��2<�E&�*��2KG=�<2�B=� �< ���,�;L7��,��<��^=��V��B�=ǝ��p�:=��9���K�ϻ�����Ȏj���=·�`GI=�Š<l�W�n=�5=Y/f;��=��0��̻�Z<.|�=N��<vh"=��i�B��<
�4��2����=�h�ݼC�=%r#�l��=љ�<�hw=a�U��g=
�p�������>�I�1>�`����<}ߧ=򄬼��[<e���n�2=��	>� ���
���0�A�B��==��4�߽��>�!�<�8����<0�<������=j;�x�H<�侼���=��a=?��m=������ͻ
�<ႚ<�n>����(�=�l���	f=A�ɼg4L�U�-9��ٽ���<�%��K�<';�tZ�=�Y[=i�<�c�<w=-�<�&=G�a�;��=�V�=��=�H<��W< ����<�_�<�E�$W�������X���s=�!�A�戴��v��/�d=Zӂ�%M���H��^�"��ԽeB�����?��������z8
>C6e�Uy�<�ژ��4̽�Ļ����>Oo:{~ż\>�H��-���]�A=ߌ��oB���H����Jټ����4��;D�#=Sv�=�x2�hA�k�߽���<&2�=������<K�*=ۉ=� �=b<<� W�1�k�r?�Ty�<Y2�=�<=��A�ݟ�bl��⛼o黓'��n��y��=x�м��=;ƭ�GX=����:��M��}ƺ=����8������7�=���케�>=�r=X�<p�:+��w�b�gì�����<�<r8[=�2�*���U	�<,�L=V�$E���DO��<w=
��<"A�<Fȹ=_�4=�w�=�v��[	�=��<�4f����m�>/=��J��;=�I<ӕ=��ý�9��#�x\����F=x��j��<�X���7����=�`��1 (�����m�=g<�=���:z>S����<�I껴4�<�����h�=��=����_���}8�C���D�۽G)��©���iN=�R����'� =8�T���U�����鶼��޼"_���k�<�k��'%���<����\�><$������`"��Z={��;!/�=טf�A0&��C¼� =ڎ�=�hg��τ=
�;���|N��Eʅ=j&>���=�T��{��r��<�a=S�}=��Q��z�U"�<,�U��W"��|:*�<��-<��e=])>�b;��l���� =;��<n���B��=M�
=�*i� "�"��<0�<>Ձ�R�<r�r���=��>�_�qX�.�=�����qʽ�� <&q{<e�2�6X��[��!($<����{�=C։=bAĽl}�� �=�|>=�
k=,�M<�?>�=t�I��Q�L=����=�g�,��o�=yŹ=Џy<�	=�P�� A�3k=�?����<�砽�:"�7����PI���d�P�=z޽������'A���3�i1���4=��J;�r=��z��Ӂ;s��=K�=��<~3�=�c��RY=������<��=ؗ=����)�ｿ�<�>=�5�<ҽ��rc��ۜ���Y���	;C�Žf�4�Y�U�����X=`��<b׻��7��Lּ�~��[_�#����{�<�������=)ͽ��`�x�=I���;�=h=�y�����a�h�;^�<<�J��W�K�u��ӻN穽���=W�8=���=����S�)>�=e�8=�`¼"�<$�=
7�<��+�S��s8�<�I�=�vp=7�S���j�=�
=��5���	�q�;�s=�潢��;/�=���=f!����/�=~��̼oc���=i缼���d=n
$�oB(=�#�<�1(���н�L=q�<i\�����'�S�	I���H�;P:==��<�	<�#�=8;aՅ;E,A��F=.�=�L�=����ǪE�Rx=�;��\,=�Ź�
��<O�.=�x伺<�A6&�k��=������<!�"���<���;W��=^�=t����`�&�=��=��=D �<3=R���%�����=�i����(==�
>Ҭ<�7=����L���X��z��a�ʚ>�7B�5j�b�{=y&�;��ʽ�K=Ad�;��#���R=����$I�����<<���Wֽ����;@̿�8��=�QO��Ca�2G0<�����'�!m=�<0뛼%ڄ��,=��<W߽�����E�=_��<)�I<)Q<���5��<����u�=��;��='��=5�Ѽ�=�<�a7=���@{�=��e����=m�<��ս��=W��<5��씏=u�=ҴN=�%��(*�� ����Z=�zp=���=u��ʋ�pƾ�u:���_�;�w���O�=H0,=B�Ƚ�%�<�/F�h��?C꼊8i��2�΅�2c>�'��A�3=���<��[��=����=�!��)<*�˽�U׽�i���e;��.�8#��fPX=�R��ʨ��#�<I$=7q��_j1��/c<�i�;bAe�1c����=�Ž2��-T���}��0��6�(��*Ҽ�_���Q=j�½2H
FStatefulPartitionedCall/mnist/fc_10/conv1d/ExpandDims_1/ReadVariableOp�
EStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@@*
dtype0*��
value��B��@@*���=��<7�ϼt��'�=�y(>q�T�8�=B49<�)Z���߼�Wh;��(����<L޽3����Xa��0���<����=�/=җ����=[r/=��>�\5)��-=�b���h;�N=�F	����#m2���(=�
�g�P�#��I�+��ay���P=7�_=��=��B�Vg�.����μ�=7~���<���(� =nֽ	�%=�*���\K������ș<��=�鐼X�V���=i�r�׃��-�<����';������p��
���<h�<��=�U;�7[��-��1�����I=�喽�s���Լ�Y�<ԃX;^!�� "�UT��'�)=�茼�`��۴Ѽ2��;j��M谻0v�<�&�<��������=�ia=}��=��;=	kL=y಻c(�<2�Ӽ��=��-:S�u�	��I�=��j��]o�E����J?�XGW����;�/=�4=��<�L	�H�{���<���<��<����;1�� Q�}��=�)�=�IJ<��]����;�Fk=��{=^E��2�<�=�u��3x�\�����T�=�.)�4�=I½��ѽ���x=zew=7> ���<�$�=�3 �?�<��>�Y��IG=�xռ�ct<^��q��;ת�;5��<�yH;�u,���5�O�l��Dj��=�=?dJ=�j��<�=C�=#��=0�P��
��E<�7<^p����=����D��=ۑʺ�,=Fۛ��U3=�����>��B��L���:�A0��8�=��p:E���IPY=�I���ذ=LQ��(ȼ%�C<���<����S>�	>��׼h(�=]�:��*��U�=�yF��`���!>�3Ǽ����fy�S^�=d��;��]=Yi�Y/����><��q<j�`=��#<�T��c�%�<�㍒��F����qU�2��O⽵ �<&�E�c|�=�L��a�=���<�|(={V�='-�g�o��u½�[G=G���R�&=�3�=g�=Kk= �＊�S�#%�� <�(U���=�T>	%�����=C|4�=ۃ��{��UW<[P��W8��4v=�.;T�>�	#�<� �=���<P��=��=v��<�b(��	�<�.��Ђ=�y�=�׆=!���!���*<u���k�=�t"=�4��<2���V����?�r/�=����l;��<,��p��;�ס�S2�=Gm����<�uA��Խ�=��ͽ��<z��7=D�5=�N="�-=T����et=�<����^��R�Z=
�=wU=�e�=�L=�*G����<L����!<��o��mR�Cs�����T@�&��;��(��*żŃ,����F?��Swӹ䂹<�F߻'Kg���<'@ ;�Pڽ�q�=*�=<�2=jB.��?�P'\;ƛ=�'ջ_R<=�^ýT���<6�����s���=�Q ��l���^�<�`󽆆)�[���f8�=�<;��2A=��>��B	=��4��+�=nν3�E=	 >0�<?�L���=�I�RͽaC�<ڕ�~Y/=lJ=��s=�����T�< .��{�;���<���<)��=yW=��.�^K7�����+/!=z������!�=oJ��d�<���=�����$R�0�� Z=Q��=yv-�^"���)�<WF��G=j=��;��<7�u��OC�|Z�@�=-��=�\��pü�]a�:~��`Ԍ;�E�;Yj����;��м�NE=�@<�i���Z�0������Ò�=�[�c�佚T=+lͼ��Ӱ}=@�I<� ����ҽ`i�=�W��0X*��t��䭢�5<��9<6º�|�<K*{=r�H��,���\�=*�������=�^�5�Ž��=~Xݽ@��<�G��k�e��;�=LZ�<�I><X��=<=�.�(�=k 0���XX̽и=�|˽&c,�[ݐ<�fM=!����*������Iԕ=�⽻�%>6�ļ�=�ށ=�	�r$=�������B�=WZ�=J�4�m�b��\1<��D=�o	=�[����ǭM�p�U�ژ�<�`>�e�=�Y��}�t=�o��8�
>�����qv�.�K�M,�=�kj�*��<��m�ϼ0��=��R>�(>�&S=o ��Ƌ�=�3;=I{ڻ�')=v��=����QBO=;z�=���'"<p뵼��=�c`>��k%���?<��h=��A�S��=@�>t?�S�{=.�=�/>>&}>�?}�Ǌ��+���|��r����)U=�[;���=�A���/$�!����r��E�C#>>k���/܌;S��=��S�=<<�=��T=�@�=����]�=�nP�O��x�<&�:�FƼl*=-`=7�����#=c�c�C�=����FD�2hL���<�r�=����'����=^L�=���=��!<�
����u"�zO==R����۽H )������3��3�i:ɈY��`=ξ�<�E�a�:KО��L?=��H=����)ۋ=�m����=}悽y'$=XZ���*��1Jw=��=�
>v_=� >�����¼�JG<�ß����
S�&�$����= ���\ʽ���oS����fĽ�P���d= ���K���(�=ސ��;ܼC��=H�:�;@����<V ۻ�����"����<�����2p`�۪\��y�<GbN���޿R��
ȽZ��A�=�b<���w�q<��s=�n�<w%����<�ю�vV�;�L<�7�<GS�<��=�L�; lܼ��7�jT=��g�0u�:��=�#>�\����=��;<��f���<E�����P==uU=�-�;v�M�/۪�S`c=G\�&���= 6�<a6ؽ�'=��D;�=�=f}C=H8*=���=.���n��=j��!��p5p=Z"��:��=	��܁<9�d��Y=�}ڻ���:�}����G��_<yRϽ�N�<v�l�#����@���=�f��3�=O�_�GE������b�<��;]m=�!=7)�;.�;=�o�<�l���P=��T=jo=�<��Y ����J=��T=%��=��%Ǜ��~�#��|�l7=���"��:�Ϣ������D�;(���H��&^�<���=�'�]*"=Y<7����$<]>��<�����)=o��]����4�M:p7>���=T<}*�����:1ޗ��[�<�ur��(����I���ֽ����<�*>��8�=9�A������vC=;�;�b��=g�����e=�=��6=�p=�k�&\.�dÂ=��޽'H�>���=S������`�� *��%f���h�����Q��`��UC�2��l.J�F�>`�r���\<@!1<;�q���� �i=�k{���</�=3�н�
��7��=�l	����b���_z>��L�㝼����װ�=p�:�d�=AS�< %_�F=�$�=�+�<�ǰ������&�+�t�\K��M=�&�<c:���㣽]|��|���X	>�{<�Ք<�:<���c�����8(=�c�=dҾ=�'����R=ScR�����Yә<�l=������=K��;!9����.=�ha����=K1�~dA;��<����rZ�$�=b�)=�]�<�'��@�=��:��� �m�'���=)��=�<=#Y���D�
��=��uN�<n/j=�|�=3,����->/�����=څ�;�k<�B=��׽L˽<M��Z�r=��>:8�=š�����=Գ>�w�=�K<> �!�Cٹ�a�����G�b����`=�I����b&�ǘ�R����@�p-�=�g�=�;�<@p�<U�V=p��<B-�=�X@>�=��M�=@���ϱy����;�ǐ�U��=.�нds�Su�='�轋\�<m�<<T��l��9��=��3<7�A՝�Е=o�����=�"��>=�17�\^=���n�<[�g��-�=�]
��4���~������<ُ��OR��`"=HԼ;X��<�S=�rc��[�=��*=+ܽ��;٪�=Xy�<K!����"=a��� �=H�N�wlh=YYh=c�=6늼��A���z=a�e������<���=؇�=��Z��6/�7!==@�����=Y������LG��A?>b��<�>=�<A*����>r1>L��쐰��%T=��4=7�<�V�<=t0�=�x�=�X�<~��=	K����I���s��:,����M��k�<��=xЂ�vi��RлE������a�=�<�����=��d=�-�����޸���:5�"AZ�꘯��/H����=�T��=!���o>a�<�n/>�:y����y�C=`��<��9� >	�
>G��=���Yi=f�[�c0�;��V��|��m����x=� �����(�<�8rX=����?Q=�=��4j=�0�;�kW�<�$=�:o=��=��@=7�g���5�X�׽nC=��=I�=�����2ս��e�S��<��=�>�<���Vꋽ#~�=�4��z�'=�Q�=�$��TeC�&=�g�<i��V@�<�n��d]=��=��,<��A<PO=>6y�@�>h��nƽ(v��5.d��y	<P==�áн���=S����3ɽ룽�β<�l�<��< �~<�����<h��ip�	L��_ *�";2�5=_ļ���;Ɖ;mNǽf���R�=Mu������1�=#Q<�,�<��b�1E��^���⽎0+>�L�� ܻ2i��S�g<�?J��v]<��/>"5��;�=�tP�,$ �_�->����:2,=��iM<>����Y�=�Ur�����O��x �����=\{�;{U�ʲ>�������������3��F��_j=/�D�H{��A�=�w�[��R����V�ؤO=&������������=)���Ż���<�K,����}��<0�%=�������hM�D�=IhV=nő=��=Ʃ<�S�=��=A�<3���ݱ��J�ʖ�<G�s���n�?�<��ؽ��(�=���;����Q&��iu=Ȅ5=mH5=sք=*-�W2o<S�=%���&H�����bR�_M=��=�ý·��"��Tɽ�Ž��*=$�����|>�ʓ�e��#��8&�}���XD�󖑽�x�'�V=T/=�~�==߽�?�����=g�!<psa�D�w=*�=�!������=WMX��>��=OR_=�|>�V�[�1>�.��4�-��h7> �ｏ�0=�m�S�S=������
>�5�<�Yҽ#u
<��=���=�/�=�8���c��t���W�~>!���';�3�8�M=*�޼����vŽ3|��ԅ
=;�<E	�e�{�'4������p=I�=��>`�\=�.x�h���K��;i�;���T�Ͻ�fQ����=�,/�Eǰ=V�<���q=i=ü�/(>�y=*�+=%Β��t?=�̼���<�>��=d5=�����O`<��pū=6it<H��=��=����h΅�����g�=OR�����=0��<&=����.�=�=�=�=�w�/��F�н�D=�A��=+5��{�<��=$�{=e��O�%>�70=�H�=��ֽH޺�0}�M�~��=M8�=� ���&>�]�mF�:�x����~��Cg��p;31=FN>�+����;(TP=�����W;ɠw=�I�<^�=��<��}��4*�l�:s#'=W�3<	 �=�<��	ݓ��&h�THk;�:㜊�a�
��e���<w�����=����	 L<9v��M�{�Q;�=�e3=Y�,�v/��J�m=�e��
�Ǽw�����y=O5���s�=uU����=v�Ϻ�0=6�ܽ?�==�;0�k0����<�Ez<"0o=?����$;�Ih�2_1=�轞과�슽�7m�E3L�v0�=q�1>>>=>�f=��%=5Ǔ;.�V=yr>��>a})=���<���}s;����r��<��>_`=�><�,�=���=7���w�=�䑽bP�=@v�#!�N��2�½r�;�G>a^��ҽ$/�=��	����=���W����A<<C�:����O����<NG[=��:�e-�*�;��<���׽�f���8�=��罿���F6ƽf�<Q�=~��� -�>��9��tƼ��f=�d=]l�=:�7= ����p���
�=Lu"=���`K���ۼ�t�=���=�C�=�?�<�?n�];���,���~2/=�ݣ=��ݼƢ��0W=�a��ޙ=�p����=@>�=M�B����<�D��M��=t�d����=��=Ѹ���ܘ=nA�=��p>��=�Z(���*=m���)���	��5to=(k/��[�������J��~��@���q�<@k�<�?v<iq�=Gz=HzH��O��.�"<栭�7�
�����Ͻ��;�<�Cu��E=�N��H27=
�(�ٵǽc��=������g�лH!Ⱥ�3i��x�<��T���<������]�!�=�H�=yLR=�n9�Tv��=5=\1=�u>���tzG���@=ø�"4�ƌ>y�{<Xb�����1p=���0�.=����H>Ͽ9Y�D<;j���(:�8l?�x�=��ʼS��:���=�fl���J�Ǔj�}��:O���m޼�� �u���o<'�=����ڢ�����;���h	=F� <E�<&aȼ�Q���O8;_�	<g�9=��'���O<5�6<�~���S�,�t�������< ��<Y�>���%� aD;�P���u��'�ݼ�u�������<�쎽p����S��<�=�<E���,�D��������=0�=�V`�5��<|<���:��=�6^����iν����J�<�*���򽽙�v��p(�H����=f=�tc<���=����A;=�!<Z0�� v=���<!P�=aU�<�U;_>�������!��=��z����( 6=��L�O����k=t��X�:OQ�=��=�%��޽��|
>�=��=q�J�4�����;>W��<+�Խ@.� �R=~ڼ��A�լݼ��;��[�A�����v=�k��u<�6<��<�d�9�Q�Ma^=�E/��gc=B$�<��L=H�\�
��=
9�=N�Z<%�8���<��q�~���ȏ���/=���=��=qgj��8�=�9ý��`=����RӠ<nu;����=��ů�P^ֽKJ�v-�����;:���j�>�<	��;~�J�f�;˹�:ℕ�#��=��>`�<q�=�X�O���1�<�ܙ=*����潅�ƼbV�=��X�e��<�F
>w7�������<�~]�r��<��ռ��;=	��;:T��y�=��;��V�=8��� ~�#4=,.={y�<[��L���= C�O��Y�<�M���|�0��V�=<��<�G�n�>4@��YD�=�&��3A��@	��&d=���<���k^d<���������-;�N�<�;��O��k�=�OӼ-d�4ǼJ��������)��� ��c�=#G�ϊ�m�N=�鷽$y= /�;�a���:a6Y�N�=��={�J�U=F�n=�J=���<o��5 D�i)=.�}=>�=��)�<R7�;�<Ohm����=c>�=P��=�k=Yσ�D�[���=���8r�r`�=�u�=�� ��O�=*�<���<�2=�����i�<v��T�)�H=�=��<���ED��N�=�V�=�;�p���\�<�=8�h=�X���f�=��I�f���5�0���M�_CR��!B<�:�=���Ķ��\ ;�*~;��"��%����*�=���7���A�=�G��Fy�=�=�<"L`���7�GC�� =N\���_<�}�>/Ǽh��<t�<�6=1����=P�X~���{^=3L3�� ��7���\<8�Z�/�Ѽ�I�]����e�f��<�5��;�Dڼ�7�<_�ڽh�I��U�H)��D�f=�rK�'��!��rw����;J��=�ꤼ��>>Ґ<���c)��r�:�
5�=�+;��.=���e�< �ս�|����b��=`/<tz=����D�=�>���l�7O��D�����=5%!=�H�=�,/=�=�3ʽ�ߡ��1Ļ5%�<��h=��=ᄘ������=.A�<���<�T������=꨽�9��
��U = �A�k9���=� .������>�M;̐.���<��ܽ�j
��R�=CVU������M���<�0ټ��<���<H
>���S�=�k޽?41��=�q=D��<��!��J��J��~|=����>S�<�j=��=�*.̽5]=I�"����=W�ռ��ѻ����Kr���=��B�<�默-��I�roc;���=����e=��J��&���r߽cz�<���=�w�8�=�E�=�Tʽ��=(�E��3�=�4;��8=�+<�N�=)�5; ]=�1C�ھ��]�=S�<=L�=���=����=0챻��4ģ=7O�<��=�wc�K��3�����<[O4=�A��˷<f
U</���f=�K���������Ļ<PbN=u����!a�Z��v��µ��
�<�ۼ5
�=�a��.h��@Ó������6�<�r9�s���$<K�>=B��<����� >B�<Y?�����=��<�v�q�����=��=�>�;I�ƽk��=�l����?4>��=3X5��>�<�=�E-<��=�F�GA��z�d=�\E=����3� �"n+�}�
=H4<��<�{=�Z�Q���<[�X�,\=z�	�-%=[IR����<���<���=�ς�8Oٻ�� �>�l�	��<�����]��~�<6�G=�˯=9	��{��=��t�fS=f5�;�D���D�;Sf�:;Jy��8)=�����G����B�������<��=���=���\�=z����n���9W X<W����K=�e>��ټ7�'�g�>��<=Ƹ�;��>�	>�0<�<{���]�gP�c���A>����>��=Z]Z>$�v��tw=�� ��Jb��Y�K"����6�G�i<I�
e�<=�]���o�7�=2����>SI�<N����;a��p���^[�QH�=��>C����=B��<a�b�*�O��.\�7C>�A������|����3��r��=�=�������<���n��<9�;\�=Q��˿ͼ�L���">\<���p�V�E��(ǽ�9� =����co���&�=�5<D�3�D���<���<��e=�ه�+aj�dI����q��ͻ=�A�Y<��#�;�f���&�����y<7�>��j��d<�����2���=��+�\n�<!�a=~�$���$=.,+�2l�=��`=�p(�Mp�)o=���L;=?[V;�=3F���')�C�=iv	��Z<p�<6��Yެ=|E��=<ɾ�$O��z�&�� J=V��;̰^=�5���3��_�=]�<z:��ԛ=^
���铽���;b�'��ۼ�!��<�)��))�=��<��=�b��k�0=Y[��F=�4P=� ��PN���z�	�=K6���=oG�;n��U�<|g�<�����7=En��FF7=~ò<�x����<�o+=��O=��=x���N�����:����1��ԉѽ�\<%��=�Z=A�����=�C�=��m���>��:��Ѽ����;~<��I�/�F=޲<p�<g�'�	��8E;6(���Q�?�]=h�'=��>O��QH���L�{;�ȸ=b�d=���=��=����&�>�0=Â�=K�#��<;>�_>c��<��(��?%��"�=`:4=��=@[�=��=���=~Yr>��b=V�K>Ǚռ�3ܺ�4��I��3�<v5z�Va����$=�_�����v#�<� �<�M��b�rBM=9��=����Y����y����ཎ]�=\���꽭׉=(�<O�F�a�a�l�]���P�Ll�	e^=c\�H@9<����M
=��<�E�����e=C�;=;����*�U��*=H�� �=Y�<_���&˽\�D��>�]&�����_=SHz�@�ͼ#J�=�"=���<��U<�Cнnč>n�ѽR��<6�?���C>�����#	>B�����?A"��=���{��<��@��>��0�աd=a��<�2=��=� >���<�p�<l�]<~os���Ӽ�J��f�"��� ��=Q���;�<F#=d-N�E�Z�ey�=���<��@�߆�����=��:<fޝ��ȧ=���=���=���<^�!.���R=�޼�`��"��c\���l����eS=�
�<?���A���.A��7=9�7�=���=+A=�I��=:��$�� ��:s��sf�mU��}��=
$~:�`P���K=�U=�P0�ߕ2=��s�����X�=Y��<V&T���bn���м`u�=.M����y�.�Uc>.i�=��=Q�޽��>"�?==	�^�9>�̪=����(L�=�;/��N��b�������D}>�ؙ=�7�=�;����=�D����=�n=�ؔ���0�'�׼� .=B$	�B�}<D��=Z���Y*��=�:��F���G>,ۻx���C.<F��<&��W�Ͻ��=Ȉz==-�3p�Q^����4=�+Q�ݞ���<<��4=Q\0�ם�<��P���;ϕ�=��>�G>?<^��#>xHc=�m��³że�=9�<rL>4�ֺp<�XH<� �pa��P�<��<�@��ޭw=��<�x=���<��=�=��>��Ѽ��*<z�G�rĽ��=VdV<ۻ
�%�p�etP��~X����;Ύ=����A�^�$l˼m,��8��~�=j��;p�_�Gkļ�d<�Ѽ�7���\A=>#]����=u䩻���<O�$�$U"�W@���D4>����R���=�\�;`�=�]�<���<=p=1G��c��;��3��'������܋��a�E!+=
5K�"�p;�+%��&u��x[=�(����F=�4�!��=UT+�Om���t��Sۤ��=��;D1ü����F=��켑��=C]��!r�@a:�LV�=cC#�(yD���޼:��<�S�<�Q)=��P<
:��w�= �E��$�<M>�<�N?<88�<w@�<�@�=k]=������K^=FFp�t8�<��걔<�%��TIB=ri'=���<�-���ap��u������V�'�5���.G�="����C�=cɜ=�sA���p=ǌi����"��Q콬�)�JUϻ\i�=�,���n=(��=�U�=��=�.�=%=Y�= �,�����!=O�=]����=k8=�Ta=׸�=��=3���J=��7=O�i===�G�����:�H�;K��<�ѽ�Z���缇�=�-�N�==��<�i,=��$=���<r��O<
D��%���r�����;�Z<%#>�k�:�>�T� ���^�<.:���,�<���=��ӣ�=R��<�(~��73����`���m>�U>��=�|�<�k=yc�<ߕ�=c�=��Z=B9*=J�=Eq�<]c�=�t�K�=(5��N
>]S�=RN��W<�}��q��!a<<�p�=����b>��7�=I��=�+=Cz+>�,�ֈ�<LВ��P�=.�P��,+=��:�꟧<������˽;S)=��;���gȋ�-P�������&�R^l;��=MW���^�]5M<v����{=d3�H��=��c=�3�;d=й;�7�5=Kh�s�����<6�Ὃ-4<9�< ����ۼx��<'��<7��=u��;��弐r��н��<�"�=.#�<�̼{�>;B� <߶=��K;}�,�?��
@���i�ض~:ڗ��&�c=����y<�'Z���=��w�=d���"�=j�X=Kh�����=�����=�]�<��s=���<�lk�����i�<���=��̼ܣb����z|�<C�_��c����<\1���SN���=銺��o�[l]<�
g<�x<�N����==�=���'ݴ��%ļo�<U#�=U<�=�A�<iR��z��P���O=h��;�ʃ�f��=D�C�'餼�=��v���<��޼��x��jU=  ����=�7��>�<�����A;�	q=�3Ҽ� �<z����_X�\��<�&���<*������E�=J��<�@�=	2S=�v�=����s���� <s�=q˼��i�5��=�x;�/�=-�=����"=Ҁ��ݽ�1G=R�U=	�=9�e�gM=��=�3��a<<(��=�>!Q;QU��GR��<sg=�z��K=#l�U�<�\�D���l��=�p)=��<}W|�:G��_K]=��Z=e�9+Z���b�<. �;VÀ;���=�C=.���>��4��M�=�f���QA=��x�<Z��<�=A=�L�=��=�h绪��7�#=|�R��߃�~*@=)t�<��9�W���%pɽ��r�V"�>�=:_�����;�u=�L8>�B`�lB=�c����=>��<\׽w$V>�=X7��y!��{%�Jְ����/3Ƚ���=J�%;��B=��;]L���z=(������=L���g+q��*>�n�	�7m��/k�<*QN���<^�:�4�l~��������=�0ν�' ��dI�򋆽%�=f>�
ý�V6��?�]�>�����c�O���i=��<�%�nS��Q�+<�䜽d�>ؽ̼q�=PZ<�5=��
�8P=�ڼ��/<�;�����';�C��Ɏ���g>��o=�"�=�ϼ��u�u�\�RAϽ	�_=u=z=���=(e��g�=���=憠�d�=��p��[:=��D>�;�:)�<��S�S��=v�޼X�=�x�<�q����<�<��=xc�= QJ>�(&���o=�x����)���j��<0��!6�<nmF�!��EN�a����$=`��=D�n=Wg��-.�<�U�=a�1<_���@�����<�u���"�<� �Z��;���=O�;tؔ���=7w�Gm!�;���_�<?��<��=L
�=��\��BJ�)(�=^���F=~�=�VS=��`�&��<x �=��󽣥�=O�f��!=,��:lV9</6=}P�;��[=?����=$x�;��C�۳�<��=�%�=h�3�����UL<#tL�a�=���t�<b�X���L��e�=��L�g����V=OU.=,��=^�P��M�=�@��A6���`=oi�:	r<���1������1��3@߼��A�����<��>$R��k��(��=^=D�q�>w}.<��k<���7�x����<c���x듽b<->r~u����=@�����c=��T�̻�Qi=�v��1~��c�޼� =u�=�I�͹=F	�S����<��r�~�=��<G-��G��{U�����9������=SO>��������i׼�H�1�k�7=���=r ὅ�j�c{A�K�;����n~
�����9��G��<��x��M�$�=0��<lH
�~�;��J=���;��(�[�j<ds������W�]=���=�{���!=��M=4�*�&��=��=4�>D�=R� ��b����@�< h.=�E�=<���s��qڼ�A���ӽ�C�=��<3��߼�������*���L�<�����=�źp�+=BG��擈�A����%=�3�<^ƥ=9�=�ċ�w�@�;L��*��Lǽ�$���hH=�:*ś>��<����o8�#�T_���=�	=�	=�x�Ō��H�Y=_ ��a =��>v&�=FM�=OM����%��<�d�;d�==;Aֻh�=�z̹����?v=���x~=���=�,>�}�=S߹��Z=}A,�>>�=��P=�ʪ;��<a�n�a�{=��<Q5�=�b>�^:��B==���"=۽��Z��$$=�v��7=��<��[jż�;��,��Pw=|�A�OH��c»��[=%��<�y>9ɜ=V�c��ey�Cy�=%d=��Ϸ��0��;J޽�<�pr=b�����<��O�1�Ὁ�(��,=cĽe�>���<��=�b��N��!�df�<b�>=��<���]/½r��>��<ʋ3=�}��>'����=���xҌ<2kn=���=8O��<��\J��	���̻U̍������S��\d��	="Gν�&m��yӽu�O=�U�<���=���=���=iᕽA$�=z�Z�9V��3���A�ڟ�=�ڒ=!����>���=�ׅ=��ٻb�r�0c��b�<�x����X=x�h=�Yн�oa=��[���G��=�Ul=�
3��#j=Y,�)��ܽz͠�h�>]�T<���=Q�.��<��kI��2�2�x���z�r�~��&��"��f=�b�=Vr&�WC�	�S=n��.mL����<������?=E�=GH��/����l=ߡԼ������"�.W=3��=�N"���j<�3�< G�=!^��i�<�μr_v�v��<�Hy��21���&= f������|�=�=I��+������Y.>8���ڽ�Q@<֟�r�<�1<!S潁�;��<��<Z�8=<�p�4>�G=��ϼL�� (<�v�]!���d�<{g��)��?j���߽e�/�<�ba=�����w����=}|��xU�=�ν@#N��$�<���+�4<qx���(8=&`��_�N<������}��=jX�����;�=eFͽ�0��i=�ݽ��=$�=�����"�kQ�<�i�=�,��KP�nj�<���sxK��M>L����b=J�=�?|���r�����C����=:�
=�A<��4�<=�H�v��=P�˼'~���OZ�1B"�=w��=����o��O�Z���@����<���>�^:�~X����=����'@���6���ث���w=�;4�Z`���b/<�X����-=z;�<��=�����%=�`�3�>[��<�K��M��]���;$»����t(�*�P��v��x��~�<lN4=3���`3f=/6="1��A�[�E�v�w~�<P<��=QH[=p���z�X���<y�w=h��<�n3<f�������3���d�E`(;��߼�iX=�9�=������#=O!H=������B���=��><r%��H=n��@݇�vr=*�����S��
�<x�<�A!����:6�ƽ��=��m����<f��ɡ�<aoɼ+���?��=�={��=.��W%+=W�	=̻�=�d�=�uu����<��ڻq�S=܁ ���T�� �<����޽���<���<!3��0�[.�����9=Z�&=o�o�*�=���=��0��W<S�/��6=�a�<� <�VĻMud=�押���<V�=������Z�jݒ;�/�=B5��O�X��<�=>0���L�Y��=���`q1<u�[<���=?�!=������=aG�ZfP����.~��j�=�Ҕ<��=E��;@�=��,=�/|��<6����/�����<O���z��e�;�]<����	b�����/�����9,	>��2�<�N'>��}<c����Ɏ>�=�=�X=L�<乽�֧��Zǽ��h���=�7�=��>�P�%>Wc�H��;%����.��D!���=KZ������c��<�I>����h�>�!���=$�ټJ�i%�>B��V�u����(>��>��-���������0e=}"���d��;���x�W똾��%='d	��L���|<�+=g��<�m�<�r%���=�v$=	 =HY��(�����,�Y">�T��EL�1��K˼3��<��0=�h�RV���=�Օ�n�"��ę;�p�_�ػ��r=.yL�P[3=��-=�P�
:����d�g��<8��<���<�=�=�ꋼ��˼�!r��ü��m�z�=[��<j�<���=��{=m�=y���65��J�a��ލ�Gb?<�<��_�YC=�}�=�_ͼ5K��B��Z�;.^����K��="3=�>D\=��*<��ļ�����)��fн~����=3p<�:�<ז}��[���(;ut<F]�fH�FN�=���<�GT�ń1=
{+=��>@p�=�T/<�(4�7���yG�˽��$=�O�2{x����`�<>�=<Z��<#N�<�]��x��Jr������	>���+>���� >��3=i��=��ݽ��=5��=�=��=Ț��o�,"!=3�<jn(��ٽ�eV����=�
=�g�=.�=�n�=���=�%� �\:c�
�?�����=ϨF=��1��:=z�=���9m�
J5�(H9=k�><�n)=/d=9�=ɲ�<S�+�	ڥ��D�=����઻>Ca<ƞ�<Z����e��x�=C!�8l=�����7e�����~>P=�=�伞]<�=U�	1	���y���=�&=���=þQ=mn�=X<��ȕ��L��9�iۺ4k޼f}�=����=�=�1��$/>ȗż�+���Y���C�ow�<��&�{��<�����)=�__=̰��%�i�a��J�F.H��<O��<��-�P��l�K=@7L<��X=1��6���y��<� � u�m(�=@�f�'�'���|�P�[����;��h=>c�:%�y�#nX��ý�r�od����=�s����q��=D�F<��鹞z�:�s�<J��Ƿ��غ-09<f\%=��Ƽw��=!��;��C��0�Y+=�ms�������=�͈�Q�=�<���=c��= ��=-�=R��=�"𽙡"=� �����i<Ǣ�<�8=�Լ� ڼ�5^=0�p<�h=�	=�m�<A㲼ފW��g{�Ӛ=Ic�:XӞ=3�1=�R;8tg=��<�*d=����=3��=�6,=*��;������:���<��}�����m�����=;Ӕ��żR�&>��=����Iӱ:졈=����x�=Pi��錼��#��Ӭ�<0��&������ۿ=��l���C=(���jM�;��Ӽ�u��z�<|Z=4;�<���=;˼u�x=�큽N�=��M���P�}sw�n��;¡�:���OU�H�����*<ꐀ����<	�}<�^�<⦓=�UT���廉?4=<e�=�#<��;�
���$ѽj>=�<�=��>����6�<j�Ƚ� 1����#�)�b��	�G9�m��ǖ�=�<f�~<�5&=�H�=y��=l�U=��x<4c���=�W-��S�� �l���U�<{�.=��)�o�=5$=�V;ݑq=H$�=�M��(7��.%=Ѓ�=k�9=��=�����$��=�;��}���H=o$�󾫼�"�<�<�$o����B�=��<٥�=���<�\<NDż��:=$�������g�<�ڽ���f=�� ��(
=�ρ="i<�{���$G<�f2����A[��WQ��t�8=���;[��=�cw=k�]�'��;G�u<�|n��hN=^�$������ۼ?�&�h�r�S7��$1=��;%�K=�<�-G��m��<S�<��\��+y��ar�*[��S�<T:��+ؼRXU�n�ǽ��E<�9>��=z^x��r��&0���	������߽��7=ò��3
��A���R=hj<[T�,�1;K.�<E@S��L=��:��?�<ڋU=ꃫ���>�#�=�{r<_o��N����oͼf��=Z3�=���;�ɼ�4��LGE<��=�;:��=bs�4��<�>�خ=�K��i��UE���B;���n��=Cw�����=��M�Ϙ�<��M����u񼑪c�)��(4+=�nܼ0�d�µ�:"��=�tm��}W����<B\J�&��Z�e=�O`�H��<N�=��d��;���%�-=�#D���9�}�5��܂<��3���⬡<m=��S;��Y�eƮ����R���o�;�Ť�6<�����ֽ7x���f��5�����&=I>
a�h��B�a���g=��(=��<�i�2�׼��I=�4�������=�}�=� ⻕�9=��x<ip��L�=4z0=�p4��&�{�=Sb=�-��k�0<�4�9
��=c[��x,�!�:��(�;��5����=/�;����T?=[��������4G�2p���u+��1�=��=�V�<�sH����=*��=��.��<�7����㽀+����=��<��	�+��-���wR�A��<�MŽ��I=������A��"��� ��\�=E�μj9����8󪏽vd\=�Xo�?�z=P�Dn=�\n��<��@��U<������=��O������FTH=���=��<~�=<�2�B�QǼ�� ��ʽ��Ƚ��	>��m��W6��F��� �ID�=�����=>����e>����i�����Ϛ���_�<�ڇ=.�i�D�9=���->Y�c��>J�A;�τ��\�=�f�=M�=�}�4��=B�����̽5���\��!�<�� >���=�ֳ=�J<
1G<�� ;\����|�=c;ѽ~�S��f�ZwD�K��@=Ƒ���`N�f�м.�=�=�J0�t?�=�\D��nf<z9C(�<}�0= 6�<Pg���=�tS�;Y���30��
x���B��������Ȉ="4=�A���=�M̽|X�<Q4U<����9=�'�=�/4��g�<�r0���>8� ;���=�j0=P:=�۸�i�8��i:n�w<"��=f�=�Op���>S��:h;C�v���M���<g=RG���``�=
�9��/̼^_�=R��	�;�����.K>X����U7=�������;�Hs���M�p����q�}�<;���=�v�l�\��~���Q� ��<��4=�q�����=a�8=F��=)�;�����N=LM�Dn1<�`�;�4�=�m�(��="u����N<���`(=G?�<�X���^<�,�=G#�=]��<=�k<����څ;�"��d�=I�;��<A�]��Xh�z��=�y�<=�=�=&�e� =�.e<��%�Vq0�9���:$�㌥<�t��0�Y94���s=?0=�y*>�V��K�=��<< CI�+�X<PX-�T���Y!��b<2<���/=������� �<�h��gTC�戴<V0�=��!=$z��aj<il�5��=HU��P�=�־��<x�νǡY�A����uE7=g$��W����d�#����<��y<�==Z�0�T9>���W�y,v<1ȵ�9DK=��<��0�Kh�D��<�(����<�1<㭴<��g<~(�fE��d%��$���<�$.��/@;�磽F�=���g�<9���ۖ{�6=���<���=�dP�4JK<a��< ܙ=j���o��]'�<��T��^�����<�ؽ�C�2q��|���ؼK&���^��I~���a�=	=�Aۼе�Q�=�a����=Q0��F����)��I=��=�y$=��0=�og�an)=uᔽg룻G��<��<��$=>"��Њ=�^L<5�=5�=4����(
=��ٽ�؏������=��5�ʊ�=��<�B=��1׆=�A���>�=�����<�U6=�Xc=��<���t�=�=�[�=�{�=W6���� �ZFҼ�D_=R�^�;p�=��;�N���5�	��<5=�<��"���]>��V=�Q=��2=��lP �G]�=laW�Z����@�=׸=�����*i�٧�=�C��� �Ҏ���5�!
2�%�u<��"=�Tż�2���k���1�=�e��(���ݼ�v��� �=����N���f>�R�u�{{�<c��<��&���H=�Q��3��ֶ���/���5j�=��e����m8s=]�	�'z�<7�r</|d������?�$>�$<����n�=�������<H�ȼ[��<tQ���G+>X�ͼ�Y�=mv��G�����=Pi޽u�y�щ��j'�� ><���=X��;3��=��=�ib=��P= �}@���f�=����9�X=�)�=�����#k=��<�=���=x
<u�l���C�9T=�!�ڗq��S��[�'��P����5*>���&��o=�}>yx=��<���=��o=\I�<��x=��<��>��=��/��p��# = �<�LA�[��=�b9I#н󧽠t�<M�<�l�<Am<
v�=&�x=Q�_� '�>񃺸�<�,=�iz;t�����R� k���Tf�P��<��<��<a�Q��樽�׼�v�=~h�=��"�<���뱼<$�<'Y�<��=K�w<tZ���y=,A@<ք��i�I=*ν�ߑ�������8<���=d�r�뀽03<.�=��f=5(1<;D���j�<T�k=T�>06$=�G��V��;�-�{���Ck��,={�#��Hi����ظ=��������=�*ҼƓN=5�S��B�!��=C���WJ�=��
=y�=�j2<����F�b=AX�=�͎=uTW=(�=I�b>ݺm�ڃ?=�t���ҽ���y�Ƚ䁣�E5� ��Ď��@�\<7�Ļx���E�<� ��`:н��	�%̈́=�m	���P���e�|�y,<nӿ�����;=@��;꿵��2����
�D�ݝ�=S�2=���=Ꜣ����p�������=O��=�$ =����g���6�J�I{x�-J6=��\��:�=��<�eR<_ﻫ~s�,_<c�0=��2��ޯ=�Y�= �">�ej= ���@�<:��lЈ�ᯢ<�(�=���=��=^��=t����C-=!I�;.3����)=�߇�
�P= �"=��x=R�W=�׋��������<Pٗ��C����6�!�g�O0>ݗj=��k;?�0=�6�=��=W� �VLm��J5��(������#=�i=�v���0���n<�S��}!�&<�;tnǽ��0�J$C��j�=�fн�T��qe=�F�������3)<	O�Ҟ�=�
�P ��D��7̽���=֛�xL�ib	��/6���t�/2�X���b���S�Բ��u�D�}T>�%̽E"�������7J��v��%�<� !=l���Ğ=߽�=�r>�q��q�=�x��:�m���.�Z���6�K<��W���W<#�=��<Y�w=�Ε=�|�;���x�ּw}�?y��9�+Õ=_g.��V��	�M<�O����x�-Up<���=��;�(�<����*�6??�����L=~&4�hMH�t�=��<9%�8��;����$bB=�]q������F_5��\��\5/�A����r&�������Z�^=b+�<��=S��=m�G=�rN=��=�O��>��7�=5�4������7��*���옼�9�V�=�]b=�{<Ԟ�<n�;�&E�<��U<����<`�=�!��������g�DP=��t�R��=�>���"�ؒr�j��L�Rc��30�=0Y����>�/��b�*����]<�B4>�/��+����<�,�4=>�ֵ=�TZ=��M��=�>8ɽ��y�L��B���e�=9�)�lU����"��*L�	1>���=G�=>A��<*6�=������Z�v�=�.1�~�!�����P8=�+���>�9( ����M�cV<ǉ=�&�<e��=S�=p����\�=�cQ�\%���G}<'����<�y���j=D��`��a#����-<�3�<��ɧ�<nDS��}��s�O��!=
M?=�Y�=�h=D5����<{e���K;��~='��=\�Q;�%=���=M�.�Ҹ�;޳���A�=>�h����<$�<�ྻ�I�<��K=s�=�>x�׼�r;=zz�Q
k�y6=��O<�t6=�,�{��a���s����a�(�,< *s�=ڇ=�%.=<�='އ�}��<:��=֤ =b��<.�h=vT<WCI=)&V��WQ���=�U=,>5=���<W�D�,����⯼rn�=�ݻ2���<>����=Wk=Q���X!d=�*�-���N�F=�m	���Q=��:�N�b=�*����=�Ŝ<�׷����Q�=��=����"��;=�1��*�<�
�=9�=���tמ=������ =���<������4=�ͽ����8;�g��]�;mμ"0<���&��FK�M!�=eq�|�=-��<D�r=j�F>��^��><=>A�����<��XEV���'�L>�z{9/�7=E���=>hp�<L�{=�J�� <�4��/�#=�#�x�x;'��<�^�;��I(>H�/�v��<R�Z=6��=���=b��=;��=��=5gH=��;Ǔ۽�S�:o��>�@�= h�=�t��R�Mr1�P;A<�n�*:Ļ�Y;i?�=c"<(�O�I�b�9!��~]�;�+>n�=bo�^��<�^>V?�z���aS2<qb�<�\��<�pǼ�Z�I���Ni>�\����<����%�����= w=f{�<�8�h��=�=~Y6=nֽ|!$�韵=�N�<����7�v O�] �=�>��=iR��<1B��?�=��ؼMd��1<� � >/.�=��C�L:���W=Ǐ����>H(L:ZM>K�=l켵�<'����Q=Z�&��~�=Y�V�o�u��I�3.Ƚ������=O����>�J=?��K�=j�=V�<`&=r�:=v?=i9�=�s�=圔<�9�"���Ƚ���=|[�b����t��Z�.m=D=�=���<����Iz�=�=E�<���l=�O
���`�<>,�=�<��;��u��=hV=7\�<���P��)=�� O=���B�C���=;����A��,H��P�<�n��'5�5L�=��	�9�G=��&��+=;׽�3�<�;»�A�=05�;oP�=�;=p_==$H˽��Q���!�+s ��л�U=@��y���M:i��<��<Bc��QA��jԫ�<`E��;��;�=*=ּ᠑��=�<`�-=������<�y�<r�P<&�==���7��<u�=�"���A�=��r<H�=X>�=�m�=��=�Ϗ�r3�����=wj�=��x=�-й���(db<����6��=&=�yL<�$)�(I=�=�5#=����7Do��݆�Ha� �8�׼f�n�<(>�딝�PD=�ͽ?:�=6��:ӝ=����=����p=M�w\��1�>"E� Z�� Df=h�˼y�><�1��SP�H{/�%����2��p�=��=T����?�Jn�=�l�=�B� �=HBӽO�Q=���g����=�!=7�f=�ý��=׆��9��=|�D=�	��s@�mJ���"=��n=��=��l;B���p�I;�+M;;�<o\ �U�</�0�#�<�O����;�:<�=3OL=�U��	��<��=ڝ<aP�;�O����<u��=�;l��E�v9�;̓;�=:^��y�0��(�^V廃�$��?=s䞻.�ͼ�[��Tu�;H�=z�<�ۘ���<���<�=3���׊=��K�b]޼f#�;rL�=.���qR�=�M��C�=_7�����?{=���=xHb=s��d�=��=�u��oG�;ۇ=
k�<�hнK�$?�=W&���i= =T�$�Co��*�=�=t93��uغCZ�O,�=L�$=�T=Tָ=?N�=ĥ�;�D=�˅�%~�=�<������((�=�w1��A_=�_�<�k<w�	=Zuq;rō=��A�\��=��#��sA���=<ü��潒��<H�K�"(=���h�����=^<��B�����<�<��y�K{M=�p������/�=�:��~�<y��<�I�<�:=��=ڎ�<f��<�̥=���<!W=�R���=O��=�˽8�;�!��@�=኱=jg�=ѿ�<B!���u=f3�<�X�=UlȺ�@:���=6�>/Z�n��Z�t=%�=��=���������@:��=jm�������p��%�=='=�:	=�E�=�K<��<,N=�ݦ=�Ӓ=�tv=Ң��*�~�)\2<8�=�MW���G=�م�
#�<�qv=A�����|kR;��=��.��ߘ�����,༸�˺��<��=(h=u���_�=v���&�Έ��4��=����x�>�M��=��)�$�=��1=@�=��μ���:9g=�-<���<�ӆ<;��af���>U=H����J���8��l��/B�=��=Lx�;�㿺fc=hKX�<} ��}<�@�=&[D=�$������|y;v��<;x��1�o=�$;�a������8�=�O:�H׈=�T�T)#�l��<�䟺��׺4X��=b���;~������aXu�}�^�ՙ5��G]=7��U	��]:=3�]=���=1�=�1N=ے=y�=v���t���qn=�<�f����=<<��+l_=a#U=j+��y`��C����=�s����=�:�=��<�|���X:ׂ=�����A�;�uO��H���]<�.'��R���f����9=��={�=%��=�\X�C��=L��=���
�u=x����3�"����<�\�=�iB;G��w(�=VK��,򭽃-�O$<o������� T���׻���)== �<��:�����R�<�������=�컽Q��;RP�йN=�=0g�=�k�=�*	>�ޅ=ӱ=~\ʼr:�<kM ��c= �MJG�&=��:�tF����=&T���?H=i����<�h�=����:)�=$�9=�m�?��Ta�=.�,�9�=㮺;r��<�c=��v�7n���=�\����=>�c����5?�.��=ü���<�*<�ļ���8j�P�|�i�K�}�׼Τ���Kϻi�W����=�����M��\��;�P�<�s�=(xF=�ԯ�"i�<#�G���:b�<�,��ŕ�L-�枊:�^o��R>��c<Γw=�t�<2�����"��孎=Q%�:4=��ɼ������=wP!=\P��� �������?3�����
<��?�Y/׼��m���=�� ��GU=�2>�K=�����p���}S=�D <*(���=g�Q��=�	��=b=��
=�1	>
=�4�<)�=���SI��I<��V<y_ =	mx��&5=��=?�l<��<���,���i�� b�;f�˼h�~:M��=�1ս
�0=,�A�#�J������W�=1�c=���=Co6>C�z��[=��=S������)�ֽ
U��
�'��|r>�7k=?�RɌ�Pϝ��4>���=-�=�<=+X=��1<��<�XB�<��=5��<$�s#�<׋�=j?G���(ʎ=�X@=P�&=�[��nJJ<�F=���;[
�=`+�=0!�;Fi����)>a->�(S>�d=��W��}=t���Ǽ�>$�)H����o�J�2>���Ѣ���+v�l�w�X5p�I� >��C�r�f�k�9=$n=L{�<g�w=q��=�%�=��/��9=q�=m�U�Fػ��u=��RrĽ�03;���,�8��	=\�)��+�=���<�\�Ґ<[��Y�<��=�{�=��l<o�R=�C����8�@1A� ��M��<�I�=mB:�s����D(�qh�<��=.��=�ç�=��=h�a����<>��<ٴ�;/]��'�<.+ȼ+�=�a+=�p��ϼ����)��b齾�=�g�=��=TRݼ�0=�ע�ob��?��eT=6.����<����;H�V��I=r�����<�߼�fe��l�w�G<4��Ȗ-��l>��Ҽ��<1��<�6��.<�H޽�O	�R߽�I������յ<Ћ��5�z=LA=��$<�&}�rǞ���m�M�O�#��l���A,��1<�<�<:#S<u����O����8=���<h�<�Y��)h�=�ȼp�#=���=|�>hE���=�<\kq<@w�<�
�;2l^=u;�<�W\�>����n�<�/$=F={�*=Vm�;��ʽ�E�:Ĉ(����;R9�B
�<�&@;�.=���=24*�wp�����=�ѻJ�>����5Q[=�?��I+�=���L��?i�����OX�;Y�<P�i���ʼ�	><(�">B'�=���Kn��e�9�7�8=�[Z=�B����&�'�=�O=ژ'�����0�=�Р��d�=��<��i�J�<>u�=���;Z���j>�E
�C|<6�{�8S���/������ ���B<����X><��=e&<Z�=4�!����F������3߻F��<��;���<�آ�� �=:,�A`��B�ּ�dѻS'ټ� ڽ��½t�=�߽u���<�X�@9N`8�
���җ��d�<��=��<��y��ҽ�n��5�}���ؐ�F7e<�Eڽ��dGK=��ӽ��z=G˻Ŕ�=�t��@E�����;�d�`�g�F�=�+���׽н��=���-R��J�����=�귽��=�ً������=�r���	�S
�=��;=^�=3�m�����濼}�4�q.�v�<2�=�^
��Z<^:5=0K����4��nټU�s=� 1<��=yI�<�_��;T=��B���5��R>,���=9�=mK��F�t=����e�<
�_��ϵ<��=��D���8=͢�=��<h�<�4��Ĩ><ֻD��</O5=X�.���1=�!�<�e��=R<�0�D�S��߽ �&����$���ϭ=�+���]=�$z=��>=�����(<�"�=���w)�=��ȼ �LL<�:<�e<p�+��޺���{s]>7��<}|�=�z�<��<��>AG=J�a=��X=��<pD�T3����>[�=<Χ=��T����<��=�w�S2=`�t;�W\�9=n�E�D��|k ��x� s2=,�>��̽i���6>g��=!0&> !�=����:����ͽ� ���������
v�<��>����j�r�뼘9=�Ȁ���/>s���	Qz=�<Ĺ뼮�{���<��=� ���2=��j= �����<B �=L1��&���^�6�RV�;[��<��� �<?-� ʼ^���[,�
��=M�>X�Ի�i'=���<='ż�罄#=����z<�胼��?�������B��&����s=��5���/��R�=^\7��g�=��=�(�=��8����ei�<ׯ��CZ������7��,q��<�����ڤ<X�=��=�Z�<�y���AмX�E=���= 8<�0�<��ּ���=Vnt��Ӽ.�0��F�=�"i�0L��^_G<�����=��^�SF1��=6�k��K(�kt"=<��!�� #�<��R=8
;�x5=�W,=�f3=U�=O��=�>M^�=�tý��<tY}� V=���;km�%�N<���Xd�=
�,= �!=���=�o����<|��=h_���=6_8=�=�[�=�ps�]Ʊ��"�h�<,��="�=���=��=�珼�$X;��=ٲ4�8�d=�y��*=�"~�U8�>�<��[�ͽ,q�<8&=ݸM�jpy<����u+m=� P<<k:����x䢻C?ͼH)A���2=�l�<_�X=��Q=Q2'<N�<C�f=o��<�+�<��;0����<�o,<@�໡����<|?|��`$=��ʼ#����L��W�;�ٟ:�u���w=��h�Ȧz��y����<���<I[ν�w�<3/���<k��e"��7(�<�nW�Iw#=f㽶Z?:L�j�!�\��	��+�<G�`;�l(=Kk��j�G=�C=	��1�%�)���S�AC ��y�=�����<���1M���t >ʟ�< >�=����>P�M�z����l��8=���=��<;q�,�ӽ�ԋ�Yٺ=�_>u��=��P<��=Ł�<Iu=f��X�����=��>8���k&�L��=��=YKB>�=�qJ>�>q��;kм�����=\(��JD8�l�?=�(���1��¬<2��"c=,�w��&�����=Zּh�=�B��Pg�9_^=MO��#�=E�[<��<��+=�=�sꇽ~<ؼi��+p5;�O�S ������y33=�.=[��<�z߼"���Jٰ=6=)J�)	����C<��T��=<�E'�6��<\ �=3�=�Xʻ��λ���;�U<=�Y�<� ���g�3Eb�|u�O�Q=�6#=V7!���p���(<ј=9��%,�-uh<��&='������-�D1�;�p;�è=���==%
>/l�;ڰ�;�����p=��]=Ge=[I�<FK�q��s���j	��V<�=]�T��A=� ӹm_[=_��<���<<T-�ĝ⼕v<-֐=sN�=(�B<yԼ�%^<�f�<�m�8��<��ƽ��������ߗ=��=P��<Ưg=��<�	*=��;�W���Ւ<A���FU����<瑐�*H��?� <\���*��;.y*=����`=�=oy߼��;=��g<��꼝�Ҽ˹�=�ۼ#�<<g�"�{�G������Y=A�<�����(��=/'��:<�o�ֲռ�_��Hc�;�#�<E�=��q�n�=��ܼ"��;��=�\=m���B�=���=�߫<{h ���p�&�.����Mi׽�x��<��=�3�<���{HU=1�N�f%踭ژ<8�Z=��<���=c�����=�!=C��������;u���uG�<����($�<���'%���=�xv�Ix�;�G�+�ּ�}�ǯ�<���<��4��=?��߫S=Y�����-�'�?�R�`=��=K@w��S�<�{��2���L<4�g�A�;=����>*=��)���8�H>>`��<����mIx>���=�̈́������)�?ؿ���(��$��-�N>�ǁ;'c�=pڼ�f�=����(Kν�g=���<2��<]'*<�==5|��m�K..>>�b��g����= ,�q0�=��=^�>E'�Q�彭#���?�=�=׼���=��K=���=�L��|�%�H������=�M=iW	= ��[�cY�=sͽ����`n�=���}e���u ���%<XG�xؽ"��<���<:O=+��;��<��6=�QJ�<X@��7>��s����~�/�����b �<Y-�=�R��8���zɽi�K=��>"n>Uꗽxf��j�=��J=��?=ヽQѤ=��=@�"�7��n�=k�ν�r>��^=��>��">�C�=p_k�)���4�=C����;=
o)���`<��J��jx�o���R��<���3׆=i߭=)�l=�P=����[|=����xn��ڶ=f2=�<�='�=ѯ�<j�� ���Z�>�'�=z���Z�<�C�=`��f˂=r���$I=�A�<p"�:9�/�_Vg��4ּ�k�%`=�Y�=��y=b;=��;y\6>�7,��9�M��R�����=�I6=�����=�M=�z�<�K�=�vܼ����c��ձ=r̜<芋=�Z=�E=��x<����%=VU	=)�ƽ��n<G�����6?�=�Ii=@�ӻ:ڦ=�iF=��B>���>��Z��1��돽��ɼdK!=[��딧����}>B�7��V�<ྒ<���E)
>̧�ș=�D˼�=Vؕ�Ġļ�ߙ=�e�<�s�<��5���|�<>lj�,�r=��z��=��=�Ä�H��<�K�� f=����	t�=5J콛����A< ��=+Ȋ>��P>\A�G��=�2�<�!��4����N�~�<�E>�Ɗ=𘆽�pQ�o�/�~�M]s>)s�=�lu��{}=BA�<�����aU=�!=���=,�+����<$��<�勹5uP�=�=�3��*ʻP=0JC������[=u�:� �ʽ<!�<8|}��a�<{�,>��<�+���5�	9�)�F���;i���(Ik��/N��%<-s6<�S=��5���Y��y�<�����j4���l=`�ż�_B�c�N<썐=�鏼�FI�D5�;-K|=�-h<s��,��;��b��yȧ���=�v<���+=&q�<,�7=�"�=�b�"����O=��E<O2�n,=va��+�&=�Z2��X��m��;Ǡ�=��e=�c�L�(��'���>�=���2�@<F���]�=)k���K�<����'�����=�P�9��)�բ�N���"�=��>G'�=@)i< � >�R�=|��<n��< �s<�>��>�{��=�<�J	=-����C�=�g=l��=��=��<l���`7��<=	Mp�
p<�����ӼS����+=�Z�J����(�c>J��<�|�=M�B<^�;�0���7x<�T=Q�%��1��*��<umA=��ԼP�v��u���x��Q=��<L�:���<��=�-s�#z=��^=^�>>�|=�gh��X=���=�u�<�"!=�yz�ĳ=4=(�s=¯�9��<�Zm�?�������x��<1�;}6^=oe����<8 J�4M��ڈ=D�r;�eл�hQ=n�)���2< ��<�4t��"Q=��ѽ�C=�|��/>य़<�=�<�t�<��6=WG<c��B�<h��<Z��<���*��<4Ō��V�=����@�7v��=b0���g��=�)^�')9���*����f�<�P=�h�����	��=uy=�NZ���;�R�=�,�=��$=��������J=�=���ae�=4�<.Ml��ן;����<i�<|9�	i�<���Y�=���=��=3�c=��=�E=i�2=J݁=��4�=�~���x,=��x��5�<L��<=vK�E������t�<g[�����<:6�;uY��彼V��=]B�=EX�fK�;����<7,Z=f��=����%=�Hz������ۃ�C/ۼ7�	�k�><JG,=�����ڼ�(F=�U?�_�&��Իb8лS��<]���0����F<%���*�=��=]��=�F����I�k�4�T���{��<��4������U��X$<3\�=/&D=F@�<�Q/=�)���'�=b
�;�����r&<��V�����㉗=��z<��ͽ*~�=�s�1
<Pv��]�=��.,�<�Y<0б<FVͺ��#��=#B�=��1=����<*�8:�M��G�+�c�p�h�����V<C�/�t] ��2=G��<E3��tS�=��<��$��g=<o���<:|�<�\�<aO>I�J<��P<�x=��y=t��;۠< �W=es��oW<���=�c�N�x=P�h<��;ju�=F�㹽��=�؅�@,�=�:�=`��=� �l[<(ֽ�'\��#�kW�=��>[�!=��p�1i=�=;��\O�k���vκ.@�=��~�)#��*D�<E	��+��2���L����=�FS=S=*��)�m�rvo��6U=�B�<���<6>������b��GW�}��<m���u#�6��m��<�3��V»�S<�Ҡ�ȃ=d:��!4<�Z�;�D8=���=�e��W�N�>��l���A=��?=!Vt=�ü�c/��b^�k���6dW=���gh�=Nւ<Y �<G㿼��<Y�������ԇ�<�g�����:���3"=�>X��~�< b=m�սuf>bM�;:Y��1}	=�#�=�f <����9=���;k�=���;"۱���:��7��7S����)�a�o��b<��7= (<=:S�<0�<�J�����!\=s�m;�2h=�`�<�I=�
=�8=�(�H���ha=���S��<�i=��<�0�<�;�j�= ����V�3gμe{ҽ��
����=���=�7�;�=<����}2f<l�ໆ��=��W<,(��w½ﰅ=G�ɣR=��;K��=?��<�i6=>>����Խ=�4�YCd���\�� �<�i<�%��[`�>	�1��!�=��<��i=;�=mѡ=�s=�_̼%�R���_=��^=�<��O��P?=��R��0C=��;��*��̧<���=�G�=�I�=~�z3���pL<>ێ=�����<D఼[̊����=�z>$�>.��=C�D�BW=�������7�e�����Y����B>�' �go��:=�#,<�!��=�4>�/нk/�<�:<`l��`�(�n
轊��;=�D=��=�=a42�m����ؼ��=D<�x=as=����z=�7�8� ;�$�=��Q�_��=_�=�JN=�=�kN=(?O=���v};<�=�Ǡ��7�!�J��Q=M��M��$\�W�����:>�:�\=�9�<D�;n�V�/?O�< �$,�v|�;�2�9=�!��Z2��
�*/=8�@<~W�;0��<��o��&=��=6A�<����#���E��l��=�v�<�Am�u�2��N�T��=>�<YW=�Q#�,��*�7=�1��zbR<b�@:el�;�7h�����FY=t�<,�Y�o�=]����[=���=,ƼANA��ܛ<VA�Uq�2��9�%=[�U�2�����<k�=��>wq;�<�ż�g
<��0=,�=�}
=�sM�/�,�}��=ٻ<�&<[W��ѵ�<�1�<�f�=�Z��rq-=v<&�$';�Yaf�{J=9�	=u�������}܉=��g�Iۋ;�	_��X=�>~J���D<3=2=�B!<c�<�A�=�7<3�,�֝��L���$s���<�������e=	�P�ܽdDf�<�`���b���E<���<�3���SF�Ǝ��٫g=Ῠ<�'.�Q_�=��=��=	HX��=������P�<?��N����y�<�����d�=9ֽ	u�<���pa��*N�=ѹ=9a�=�1b=@>���=F��=�����H�_D�=;�2>}�=	��נȽ��=2.>���<�!=ƴ�;k;�<�)��家<��	�H���B���Y��V�:���=���ww;
�Q���w�B$�=d�<�p�;�F��q�<ɚs��<�n�<��H=��
�̬2���{<x�=iU��W��;�El=�1$=0���:>��l����<�z��㱽U*=�q�=���A��;4��=�m�=��*;	&~=�࿽�gĽa�=�)_�v�<f�D�r�y=Mŀ=��<"�p�~/Q�H�;TM{��=�����='Ep=��%>�l8��~z���ܼ\�<�;�<�<w:r:>�<�m>��ڽm�������=}%I<�?����;,\d��"S<����U�=.�=�H��]�<h�	�[��<U�;D& ;8U�=%��=��<�������T;�:).�<0�����>F�*����;�P�=@�v=�-�7p>��k�e=.�=�y=Oׇ�*�=�7>m�D�}�n<_`�_ʽ%��;r���=pP��1Y�T&a=�t�<�y�\=��<�1��ߺ9=�$�=���:��=��i�8ի<��i=ub�<���=�尿Q�b<�(��p����B�Yˤ�.�+=O������ d�=�l�=Z�_��x�<�޽ߨ��&w=���=_'�<��߼c3�=�*�=���J��=��=���V%��1^��8i�`����+'>/��:���T$<�N�=���=QP���E:=R{=�8{�^��Jx0=KA�=J�=�6 >fh"=O������=Q�97^=�G�=D}��ؐ2��ͳ<��I<��<8lP;�D缭D�!����o=9�B=�I�c�<��>�@GZ=��ʽ~ú=pʺ���<N=d셼qR�<$�	L�<�=M=��=2-�=(h��.=P���>;����� y�[~����</=	hM=���=U=B5��0����=����+'�73&���P����=<�x��*�=�۵=��z={v�<��}���n����<�Q;����O9:��W�!<���=~�=���<R	����<��8�>5�=�5���ܒ=>Z�<;
��ڋx;��_=U_$��멽kYf�}Ȟ�@=�{ֽ�,�<G�d=gR���O＃��=�t�=�޽�]�=�I�=����>�u�	nԼR#>Z]ڼ�Q�=q��`]R=�Q3��V���(�K��o;Ѿ�=����=8�=�揻�D<m�9������	:=����K-<5R�=Y�<�I�<߁:=+�ǽi��=��=�U�<w�H<�<,<�67>+\W<�7%�Pc+=�β;)�G=�0=]bH<Gu��I��H�R����;�pϼ���[�=9o{�"7Z=.X�k߲=㗽�1<�����c�<9�g��c��w�<�ٰ�!Y<g��w~�:�.��c�<;�>κ������$={tD=���<���<Ņ�<��=3�=�^ּ0S��W�<���;�+�M�y=Ŝݼ4a��S�:�N�T<ċ�σ�<?Q��5�@�5�<o;5�켍�<8�W<��P�$|�<l�=[J����[�<��<��>�3�=$�9���O�'5��8=����Q����=O:�7�ȼ��׽zF�?�A�*�;�]�����=Hv,�����k�ѫռ`]�(�=�t��B�=Q�˼D�v�=]�����=jm�<�s�=?=��D������<����~��=�B�9p�� 9�QP�=�����<0��v(=�7�U#$<��`<���<�:.�%��`'=Z=伾&�=�-S;�=�QX<�ܣ����=�-=��ҺƑ">��ҽ^���P�:����w�o�����v_�P��<�?]��yC�g܆�,_=8�>T���&]�=�I<��q=)m;�Vu=�~�� <Ed`=�*��5?A�|�ҽ�Bؽb�= >�;=�&>�3�+�l���=I^�<�������;�i�=��<����9D�>[�n4.<�4�<�/>4��=]e��cSi<w�<�hb>CY˽���=]7��w����<��/���$��z������@>ơT�Ȏ���>H[�b.����#�"�)<������=R���1f�=.��<�v<\ɼG�ٽ��ۻ3����<����dqX��Q��?�4��!%��*>h�F=6��n(>�Շ�9g,�N�[=���<�؅=��=�<�<v_�F��=(�,��Gf=���9����-=^�*=�25�<�=<�ȼ�/�=��罸&L����=�Oټ3&�=�rQ=ܣ�=��=HE=�����S��^�=cuڼd2-:�$��O����K�=Zu�=�%`���8=S��;}���m�=� �=f���o1�<�$I�Js���ɽ�
,���<���=��/���X<�x=�2�#|�:��,��Kg=�3����<h(-<���e�<��=��N<=���Ŕ=���;��G��)ļn�<�1�<�a��[�hx�:N哽�΃=`��<x*�Z�<ig�<F�=���"����C��=؅=�r$�u�(=ɑ{��ѡ���'�u�?=�^ٻ�϶=+I��t�4��������=#��<�k��&=��=Z
��Q�<LUD��i�=ҶM�X@=:m��K=,�w<��<@���
�=ˁ=�����P�<�~�=�{�=S%=�<��~��M�g=����y�=�Z=	o�ZԂ;5~<7�=V�=�u�<�Ũ��� =�)���@���XX�й4=���^=W=�Ґ�ŔK=�۽9�!�	��4�=p7��G��==t	=QZ<�=�wV�=C�L����;��a�I=�=��5~�<�Y�$�<�(����Ԧ+�W�$=��B=F��<p��;��==�=�<u��=F�C��P��h9�}�O=��<�I���΅��y�Yv�����@�=����0��� ��M�=���<py�;��<�A"=}�-��jy���\��ݐ=N��=�&�=��.9ARQ��=5�ֽ��9�#`�y��=[����v�m�<�s����;��=,�Z��ؿ<X伤]=�t=3��=�\<;�.=d�U�!^r��0�;�t�=��=D�B<�IS�X䊼)�=By�=���=aYϼ��s��;�Ƌ��&��<q=Q2�=����Fw=Gd$=� �=�흽�Q<�-��=ȥ����=��:�{}ӽ�$��iy������A=>"�����}�0��ZC=l@�;@e<>�ƽ�<M=ǐ�;L�=�=z�=]
�����:�����J�;�&�=�\7;�8��}�Y=
�h=s;
�z�9���
<�~>=����=P;�=4É=ȸ�=��C=����:e�5=�x�=X!=X =���$��=g�<ܜ�=2�=W�����׽�Uv���;u�8)Ƚ������*)��X:{=I�<�P� �j��;:`��6P>l��<v�X=b�=�e�=߈z�����@5�H����ݗ�p�����;�~%�{U�����=I>>{R���<�6P̼�c��d��=�e�<��=(�R=� /=��2=1�䇌>=���?�<����Db�Z">���;��8��+ٽ^}�>���x�=������ �e�y��(��5�����U�f ����=�B;�����f�:}�x�\�����a��=���z�b���/_�<R�&;Z_6��쫽�P׽�⨻��+=��>�ڽB	B�V��;��<��=D�<Ա�=&Eq�v%�=�伽H8S�yE�:T`�<SԵ����_���\<��i;+ߨ��N�<\���	���x�=��t�������<Lʋ=�	=��<�ѽ~8D<��=����<z>%���,C=y�b=r՟<O>}�V5=G@�<��<�I<=�X�:��<0��=QE�=V��=���<z5����=��:�������j�*<e)�<�0=3�м�>*	��Fg�;�ɼ��׼���<d9�B�L=E�����9;�����&7��V�ne����,�l�
�u�8��T���l�=
��=x�=�j�<X��J��=}�S�5+,��=��<�H#�=#�����P�<�:`�k��R�=E���$�>Ws�=�������;ש��5=wƌ��1O>q�b�	ީ=�=d��;����5�����=�E>;6�<v��<���<��仞=Z�G�v��� ;zJ�!=�1x=(Ӑ�ʻT=�,=�ǎ=W[Ƚ7��<ʇ�<Gs=�S��>�q=����H�;m�%�8�=5䔼�k����=���`����=���<�P=A��=I�=[�>�z%�:�Ԭ����<燔<��\=�
�=5E�ǘ�=	����=<
��?<P?սM2��O=+��9|>�ќ����=HD��B� ��6N��O���`�=�y�<M<��&��M�͛�=�FT:��A����<P�(��z�=�E�����Ul���:�<Iѽ�/��K�>����=��s=�
B=��Q<9�N�({����<@>�u=L�8�ؑ�î�����<^�u�V�z�6>*qX�O=e���"ѽ`j��[1�=az�;�н�������<��7�䱪�� �=�}�=�1M�#�=*�F���>O!ѽd�Y?���=�Ef;%�޽ƅ:=���=�S>���=�T�=ޟ�=3��;Q*E<��=�=�zܸ��F�</��<�Bżfȼ��4�T�^�q����-C���<ַ�=liZ=�
�慊�֒�<�ڃ��±���7��HX<��I�E={� �F�q=�ܭ=���=h��=�2���^�jã=h��<'=��<�Ɂ<j7m�䶥��l>��?���3�L���ٻ0-��H����T�IG����;���<�h��D�Q=� �=�+��3i�=B*���<��ļu6r��0�P��<ۅ��kt=�����=�H��E	>(:�=�w=X�=� ����L�U��|�(*ֽ?=�;�����;h��=ԝ =�Z���fy<�e�
<�=*H<:=�^���f=�_޼x�D�NU���l.y���o�C���8�Ƚ(x����=��8>��=6�!=I�)�b��:�O�<�[`=�����B=�mA��	>�:a��0>(��=s�=[���6�=���=�='���սc�J>�*���j>��i��Jź��RԽeh���<������=�K����=�6]<�f� T3�<��=q;�����}���R��< �'�6Q<eH=����\$�q��r~�<3����H��
>���<a%�=ر���4�l)k��=�b<�`ѽ5��<��:��a=�#Y���"�>����+���b<�5;�di�==�!<�`V=ƃ=�'="�(��½�5>(���ݧ�=,�%��;=w�>�Q���)=� =:g�=vL$=\�>���.R<�=���tF%����<K�[�� �<���<����5!�=�`"����������<����k>�'=`���;� ='��� ����4<�D���;+{,�NG$=��=y�.��� �c�=.��;
��<)��<D^��N#���
��G�9�:)>��>=�vx=�2�<~�@;��:��-=7*=�.�<�Ɗ<��8;�+��k��<Rц=���dP|=7���)z����.;��p�y>���?=�x;=���6Q=�T_����<���=Ԑ��sܼЋ���?���t4<D�=��=��=�ҽ�m=��=Hw�/�ɽh�L�;2(<�	�Ǚ(���,>,k�����a	@=6�4�$�6���<��=c~D�8>RG�:q�ļFy����"<S��<kC｝?
�x���-���\.=Pl�>��>"�<��νyB3>J��6:tp�:��P=/S>7�i�6���Q�=�s���=�A¹7	Y>]�=�3�=!����ż6D>>y���yH����������6�����ν���;���׬0>�8��=�9�=�y�z�ƽM�=`����.��F�<Pҽ���U;����M>Ք����������dw���-�=f[0���
>?�:>��\�J���&'���]�<��$=C����ý*0��-%w�8��=���=�2=�ýi�����<Ȕ�=R�����d<��<<� =JB��+��H!>��̽��<���z�n=���=R5�=F �=A��`�:>�\"��)�=3�ǽ����Y���*(����uZ�=7��2�%>��8�5K�=*�=���<�(�����<�<�=�?��/3�;��s�@@2=Mw=�1�=��������;F�hY�<�H	��p�<���<̆�j$�;C�O�X�<oBy=|<#�n=3��=&b���=P�U=;�=z�]��[�=�=��ټ��0�0/ýErD=��=AgB=];��Ʒ�@��=[1e=
�=�'�={?�< �.���8
�;��e<����r��=�w�<G0><�R��y+�=�H>��=�ޠ<�ҽP^�<�;<Y��=��
=�q����@�^�z8<S�=�-������z����*=�V=��G��.1�(&
=�8��Ӻ;)r�;w��<����#����r!)<����H ����<�Ψ=$Ϻ�Tǻ,]=�����=��
=N�<���KF,���L=�]��^a���u��@�D�S����i<�A�<�0�S4�<ϖN=�'�6�1�n-�<��������+�<���=OZ�̱�;ڧX=@�<�)&� �<�˼��<�
�<�F���RB>>RÚ�V80��<<�����=�=�r
���W�ڀ2='��<�����0;.�,��ż�ݻD�>hX��ԁ=و���<�l;c��EUZ=� =���E\�<-h��˗U=�o�"�<���@�=H!
=�,�$�y<���IO-=�ȶ�f=s�ڑ2=��%=?!�Oݚ<��۽��<��=M,�<�ܩ=�=�=�×<[�l=�`�<����q�U��Į�+\=��}�i�|���W�O�,<���;V�Լw�b�� m�քܼ�+�<�C���w�<B��V<?�y�W�=� ,�?���V���3�\��^�=��Ž�U��02=[X���<Czm=!/=��i�7=Ov=�ě<N7'=�<���M=���=��';9m����=z��;��<t��<��R<�ǂ=)9�<Q ;�
�<�B��"ټn"���N�{3=ͥ��0��a`�=��<7�;<�1�=��|=�m=���ɻġ�=���*u= �=9��<�@<
cG=o�W�v��:����m����;����=�8=u8عFk(<�,�=�1�陽 D��;�n=(��� >	[$�G���i��=1��e���L�<6WY=��j�{�w=�:�<��@<�"+<eL�^Ґ=�/�������
�=#��<F,��-�\<�����=4�<PГ�V`½T�]=ќ>W�⼰Ģ�K<����={�=/��=��T=
��=�Me���=��=6E�_4/=��,�=���q�<��<U�=b���������~=0Q�x �<`���E6�=�4��'!w�ڐӽjE\�r
M<Ϣ=n������[�<�o�=T�̼G9���d�=+����A������D=�)����b=WoO=)g�f]��*m�����p�=}�����=���H�ۼ�+=��2�+��=݁���=!>�;����W�=���=�$����>��ԼO���>)Dt=~^;8R=�b�=����=\�v���<��2=�hL=�ɏ<�P�<�����y�=��<�	>�݄;�żb����֕��w<���:�n�<ӗ�=u[|=��d=�.�\dX=Ab�=��<ܙ/��|z��ښ=�.h��2�=��)�'=+s�=
M������˭ݼ��G=�y�����=���<}6�=���<?㘽_"=B�$��c��'�;2�;�s8����Nl�=@�����'	�<����f��G7��s�_��=���=2�=�`I�hZW=>�����~�J��<߮=�co= ��	s�?�;���{��_dZ=���maU�@0�=�k(=TD�=�+�<��S�a���ΰ�CS�<@�(�W
�d�=�»z���e\������n�����G���>Q�<����v=�$���ֽŧ�<�}1=	2�=."���ƚ��. =�q<��<d�&;|&�<�_��`��=���;�=}���"=���=�U�=ʜ>�h��棽�%�=wH�<"M=��瘼�.�&S�=��Z���O�	~�=?0��Օ^=I$=�BE;(��=�k�=D��<@.�<h[-�ē�� �;��ԕ<gs=���k�̼ס<�n����<���=%3��B��T�=�7<��<-��<���=��˺���<��R��{�cto<�.�=�lϽ�׼�Vü�U��	޺,b�<���@d�1j�=��=�1���E�=�C<�t3<E��E]ݽS4�:��=��ٹ1*=m�h�;�<�'�<��U=q�O=�>��F�<k=���<x�4�Fϖ��?ƽ�Q =��=���<B=��zǻ�R[=M>��=-�-�c=~`h=�?=:��(�=���L=4�<�ް=a�"<�Fｐ=潫��Id�=	��,s*<w(= f�;�6<��>=�O�"���=;'��f�9�";=_6�=�<�I�<�iO=�`=Gn��;ܠ���<:�.=NZg�,��<���G�ĥ�Ii�=��=�|�=y�������w�=HF�<�!�<}�=Ŗw=�4���K�~��c> 0=4k =����S����߽ � =d�ɽ�T�=~s�=F3ܽY��=ռ�=Ez ��8�ʠ��uU<���<�8H�sFE>EJ�ə�<!��< G=��O��R��:�'>G����5>��(�A�Ǽo�/��A޽ڏ�;_�#���k)���ν~u=13>�V>�KN=��A��ڦ>۩�n`��X�H����=�%�=ժ��s����0�=$}��>�t�=�C>�=gq<��=��)<YI�ڢS���O��
����<�{�6���%�l���E�;0��>Y��;U�=�p�<�7<S𗽔'M��o>�J���$��nh�&c���S,=�S�=;�o� ��=�4F;2���ճ=L��n��z�<��B�<d5�<���=��~�4�=m��<�H�<�!�<�V<�v�<��=ȟ�:x"��Tq0=�҇���=�<ễ�Ի�w�=�"G=U��E�=�T+�z[_���D����H�O�rf>��<�Z�=�>z��=+2��S8�=-;�����<_n~����<������;�L�&=�(�=�<���.w&=��`��꼔��<�Ub�xM=\&d�H-1���=�\M:�T��n�=��C�B�<��=�ة<t-=\D=���<i�8��)�F[鼷,����r��#�=־�=xB/<��k�ȆI�3e]���{�f��;a�=y-���x��!���;L��~
�������lf�
6�={�<�����=����5��FW�<�I�<V��=mi:=�t�;�==��޼���=���2F=zDP��0�9'�k<߈���n�=l<v<�H��d��o�����@������6֓��t*=������=��/��p�=�ӵ=�F���c=&O�����U ��=�w��u.=����ᩖ��vF���=��+=�e��(�z=�&=Ԩ���G$�+��<\��;k�X=xJ�n����/�s����s='�#�o�$>�
��� =7J�+�<�U)=�`��h�=
83� ��>����=>L�<C���?� [��ђ�����x3���%��>�=�I��{�%=�>����4�S���<��W�i�L�φK�`G'=X'��N>�ȿ=RW:�	2d�::$��WW��P�=�F>���=����<b����Ӈ�Y�9������=-B�=�=����/<������I<�F9w#Խ[���s,P�k���.ƽ�K�l�=`E;=Ӵ��<�U���[
>��pD�}Yp=@u�}[</�����=SE�=s,>�F����!����ZC#��&��2�=�;9�����ç��N�=�h�D����~�=�V)�-z�=�����1�=���jR����=6�=W@���x=M�5:���<3��y���J[=O�3=��u�y��=#�<Wv۽ЩH�e5������1<]�E=V�s�վc<�H����-=	����<*P�^��L1缨���ػz<��;��μ��&=�QӼG���֖�=�=�<�4�=Tx4=���=AB�;t��=˨�<�R"=���=C��<��<b�6�.���h�=e���(̼%/�=��X�7m>�n=:/�!j ��Я�Y��<:��G���s@>m��a�h��Gf<�~溤)�~b�����=��ڽ��>b�=#W�=P����+�����"3�{�#=�H�H�=�>�>ċ�{�伥� >�W����^�Ԝ=�)/=x�=z���`%�M)����x<��=���=�t�=��=FR<=^�<���P��=��W���%�%0;��P|�4�U<	M�kS ��#l<��'>�d=E�@�dq���&#��Al��5<.�=�=�=��<�5=�ӑ�lB�<j˝<�Ԉ����t�pu���3=�aI�ޚ�a+Խ�����v��fn =�M8=�#�^�>=�����6<�i�<e����=���=y
�/�L=D�?<�&�E���9̽��e��~�<�,�;_��<De�=(�=��M,�=X�Ľ3��4�q=��n=3\�=���=p�
>�<v������]�\���R
�<-d@=����`����6	?<�i�=�h��=p��<4,�����<�C�$�+�d1�=X�e;��ߺ"z��䛽�ǥ��{�<�ǉ�Z�<���=�����=�����(<�[J��4;j��^n�<B�	=�r�`��R����;�0I=Ɋ����]=��=V�?���<���P�_<���=*4�<|��<��<���<Eƃ=#���jK�i� ��<?�q=���>�<_{�=ެ'=��v=~�Y��U�<�u�=!e�<���ɣ�=Lg���>�.��=Rf=fj��Q���2�̪�=���<�IH��o��mڳ=(���i�=�=�̂��M��`=�z�;����j	9�ư<M7=��=)����=I����<V���K@�܋m��fH��	U=p.�;գ<fCq<D�=�[�:^�;z{>GG{<����=�Ģ�����n���I�����<��<Fڋ:�A�)o�����HuD�j�=��:�2��<f�=\�;�_��y1=�H���=TI<����ͼ��0��,(0=���;
j��k��~˺k�̼oA'��}=�]�<ݓ��5��?�,>��h��׽���x��#h�=>���k�=��&�=��Ǽ&=f���a^<���=a(½�� �׶P�����|Ҽ�E=�6��0[5:'hL=���f@>>k�m���^�Ӗ���I;��i=]���(M>A�/�e#=蹼�%�=�C�=ʹ��V�v=I5O�d�t><)q�~}X=�f�Gj�<K��E:���۽t��R�I>��4�=��G�H�\�^�����R�K�Y��ܽr��=���<gݮ���d���F=��<��<��E��oV�A�=�=%�0=�R������z=	�!\�<�YļE:h���.=�~�<Ad���9<4|=N��4=�I�ԑm<�B�<'M���U���;��@����'o�C��;O�<H�&=D�F�����	��=��<�Ό��c=̽,�Pw�<�}��e�#<�(2���=���o�a���h�)K�<�溼HJ=GdD�N�=S�9=ŝ���N�&��=����[$�]ۃ=tC=k@��׹���Ի�,>=�A;�[=rE<߅w=��q<b��W�=�3���/
����<��<(8=��3=$l:
CV=6����k"�z�
<:)�=t\<P�*���Z�(� =���=R�d��>=�7D<��j���<�F��S*���R��9"�R�<��K=�m��>0�T��\ =V�V9uX��3�ý��;!σ='(o<M_��!�<�6=�%�T�<'�=�ŉ=-��<��>6�<g�ʼ  ��v���� :=�Ļ�Fl<3·=�}'�Y�;����8q''��]��%=����b�<�=����^�[>�挻S?�����<�Df���C=���=�%Լ�H����;�A�<�G��h�+=M��=ɘ#=ɉs��>=�W��8̳;b�!�*I��1�=�@=�ɽѭh�o��������<�(ټ;����jJ�Z�<=˼�!�������.=�<�CZ<]-'9�	��K*�jR�=��=�s�;ߤ=X�\g���Eν�w='��>{k�x` ;�M���/>�m�� u���}b;2�2:��:-=��<�e���v��I��=�_H<�Ϟ;�ļ"=��F���$�V�}(�ȏ�=�eҽ���}��=/�>J0м2c5=���;��w��q=�+�� ��= �=�����bI<@&=���<y�=5�<���=9u�=��=��=1���vv�����} Z<մ���9#>�
>�<\�9"��F��7I��C����<�Qo�P����f�{���d��c��<���>m�e:1������;E!��M�>g`N���S�W�J���[>w褽p$B�=7���:��ܫ<V���'��˽۞��c�R<*��=�y<�-����R=��8��>Φ��l�(�,xA�p��. =*�w�/π>W;H�*�2�Gَ����=J�=gQ=X��=���P>����S>#��D�[=}����;�T���n<�@]�29>A�h=��n���=%�:K��ů����.A=h|j=A!!��B=�y�;�r�=fA���hý]p��
IB�ji~�f���֭��,�1���@�ط�=K`��gt�<�/H=����4�=��(�"�;���ap��wy=Q�F���/�Ƚ�=
>�Q�=2H�=a�Ƚ{����=bs=���=�?� ff��:�X����>v��<��:x"@=�\$>�ٍ��˦=�ɛ�y�<rk�=��Ѽ=%r=R�F=��_�jv�<�r��:&<,]�=l���[�<��>��F��˼Ex���սG$º杊�+yP>+)�tV����:����"G�(s��g� =�c��+>�_+=6�ýjv��񲹽A@
<'�޽�޽�G������	M>�a>n�X>M���񤽏ŋ=�3������/��:~��<�q�=���|�����>>/�����=���=��]>���=mWA=�K`�G�+���/>;���~܁<1~j�Ὤ�#;D� �\N=�e׽�C�>#P�=W�����߼��z��~K�:�<�j_�W��w0���h�=P�	=������T=AVۻi�<��x��*�=��켩������K�ߏмx������<R~=���=&)<%�����=3\=��Q= �4=z�U�+3x���<�?*=���%G��9����=x�,=�.ýZn�<勺��=�O�<O?��G��!i�=E=R=���=�ʽ=�.>�y�<��<E:ݽ�<Z�<h����7������Z��P߄=O��<Cι=���<\⡽-1��K�0��𗼽#��!�{�X�I�;�Z����<�WV=:����*�TN����=�5��X��o\{��~r=�-�;��<A��=�'3;�=Լ�O=��	�o1b��z���Q=P�;��R�,f���S�.�=h�	��6�ґV=���=��`=�5�<Mփ>����<�������=1>͠��ú=�5Y���U>eU��>���sN���^;����v;K��q��,��1|=�ː<%X��92/>pa�� ��귇��Y�=�Y<8E�=T�Իyv<��k=�
>�S����U���z="��<�����*��"����b��=h��0�=*By��v��r�(���+������1C=TWX<  ��=h��=�(=k�#=9q��Xe�=M���vl<-�=���=�����=Ⱦ(=�n�;>6�=�y���ƽ�NP=QԔ=�>�#�=��=?,��$�=><�����0>��	j�<縞<LLü��=��<��L�t��=*M�<J���ٗ=�~�=�/�.�=�E���j��e�S��<�||< F��W�=�|=[�<����xT���<M@�<�K=����p=j����TS);=�k=�2��bXt�8�;���<,�>��fA=�w�=:�d;����=u��;�n
�`I���@7-<f��R�/=n�
K�=F=O��=���Q�h���=��+��!}<�p�=���N*��"}(���<=�z�=�'=`����r�`p><E�9}ސ=�o��>��//�<��<��=���ڥJ��T=!�$AE=�"G=�7�<�B���%��''��x�<,4p���L���<r�>����P9��j�=o&
�7��<�l�;~��_��������=L� �[�=�'/=� 3�-��=�+�c�<*��=2���ꮼ�	λ�ڮ�CG�i��S�&>I0�uܜ=O�=�ߖ=�؍<�$׼��=O%f=��H<D��5�<D�d=_���]6=������< g="X�<�͉��@����<l�	<���<(��#��G-�=��<Ō��Nc�<L!!�Ї��{�!�M=� x���.=��MF�)g���&�=*���5鼾��=�ߏ�#��<� ��c>��}=�h���^�<��<�Te��7�Hծ=�.�����C_�<�+��ڶ���2=-F=������=��=��ʼ�}|�?m���żC�����=N=�B+=|��=k_=
�5��{;'=.�H���;d�=ʺ��؊=�m�<�0t�D꽺c�s=���<)�潶~��]�q<#��=��=�B>;���<7��|4�B����r��<��h=;�>@ͼ�RR��|<0�=Za!<[���3?9��v=i���j�Ͻ���4:"�Α�;���=��;���=�	'���[�XG >�&�=��=9�;����<�81�ܣ���������d�5=W��*:E=N;u<��ֻgs#=-�ͻ���;Y���e��<��Z;i�$�RD<pdw<�����+S���Ȼ�[����`/=(��=���=�(B�4���	]<�1="�n=�2�{H���%=��{;�FH�%>����I�#��< ݒ���S�c�����5���U=���;�3=o���j�\� �=,��<�Ň��:s��q�����ͺ!<[�>|��}�i=��˽��P�¼zѽt�==:�<M����$ =�T�=Z팼ˣ�=F�����6S�=o�<M>H��=c4�<buû �<�k��o����m�+2"�3�A>�A���B�!�T=正��ި=͡�=�J�%��=>3F>��Y��= =�A�ֽJyK;;.��ȡ=9�)�O����'���<)m
�HXŽF�=�����=Lť<'���\ż��(�W<%+�*���;��ح뽩!->��7>�E>�D�U���w>$3�p�F��<Qx�=,��=�C=�2�n2?=�Z��">`L=�qP>Ek�<N�=�z����K ��MC�X��ӗ^=�6��k����BI���3<"s=if>n�>e��q�2=R�p;}�<�c�9>T=�D�=�b�=,ɡ����=՜���(=s[�<��N=��=��q<-.l�������<�w����e��`=H4��J=��&���&<�$�һ�=$�ͼ����p��5
=��.�V#�=ն��3�w=#�E:�B=�g�=%��q�<I��<!V3=H�1����<�ru����6$���4?=7�q=�!A=�<(�g���>q@�:��R<�(�<�0e<HA�=D�ouۼ ] ���¼���<ź�o��e�z���1�n<��̻�-�<F�;q}��q�<�=��=�����Ƽ�|���ם=q�ͽL`��9'ͽF�l;+J'<���<_��=���̺�=����ý�Ln<yH���u˼ȴL=H��<߯���U�=�"�<��N��ˀ���=���=�n#��s4���<�r�=	�H�n�z�mR��6K=�C�=�]<���=F��=���=x���bKK=�,�9�d�pL�=D��=����=1+���<�^�2G
EStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp��
EStatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@ *
dtype0*��
value��B��@ *��)��2*�;��;��	�W�-=�E����������=�g��|k���=�� =���tiƼ\^
��f���=�ѱ<��6��<�o6=��=�� *F���f��	\=�t�<�p����&=c���A7�=�=�}ʼm�2��Wv�l��=&�"���˽L�>uߢ;���=�L���a<�#���G=j�=�2=F|k���};b⍽��= ��=�7=���=����_ܺ�4b��Nx=�0��`���=���#��=M��F�:�^=j��x����e*�lּi��<�]= �=0�E���>ډ8=�-ܼ�<��=;�;�=�l���<^�Q�U�M��=�GF=x��j0ݻ�8�;*`�
T�=�7�=%��<�Pz��̄�_=G=}�N��������ӛ=�|=������<��[<�];3f��>�j����<M?<��¼��<��>1���7�<�V�����`}=�c�=G	5�1��j?�:�S���=��>W׌�~�̽�b��!0N��/�n=��"��_�<"U��L�
�EX廠������m=��^<[��;R7�<��Z=y�=ީ�=�-�~���5o��\<����3»�m=kc�����V�ʼ
�����>eϬ<!��F����;u�罚>������<
�-�����b=t&3=o1����$���a�ϪG><y+Q�s�<If�<����� ����	�=�7=����*�g�~�(��g��N�<��+<B=����Fx4���"�i�ܽ{1=�2�2"S�\͖=���?��=N>��)=���W���t8=��[�T�=�5��^w��}�<\0g��(�=C�=V��6�Z���9=�ȹk����׼(l�<,���*�=!i=���}3=ԱZ������Sٽ��|��y=e�;��ƍ�H'�<�Q��8F�>�;�׉�-J8=�M=�(<'�<�nD=W�=����D֩���,>A)\=�N=�&�=�oi��ʼ�k=j�f�M=N}y�Fs�<`����f���9=��%=��5=g2�ɭ���dԽ�x��㨽A\��hH�<���x	;��:<"���ը�/�_=� ;��Ƚ]Ž{�@=�)�==u3=K��<  �<�Y�����<�u"=����/N=>�!=/N<;xJ'���켂p��q��r6?�:H�gP����<J�����<<Mb�<A��=*q��@T\<g v=Ŀ�<�|��╼���=�+�;ȯ=�w����<4#�=�����j0���:(�h�gz�l�;�P <�B�=��G�YE0=q�g<S�*�]�<W�K<w���I=f��uZ>~��D�= ᄽ|]�����������=�Mc��T=�D��Q=�D{���/<D�i��>�=��S�穩<���Z`<K{-��@�<{Fq;џ��;�:!F=W[��8Wi���>q�e��Ǵ=P`�=o��<�����5��y�|<[o=�9�<�AǼ#�<��Q<2�B�������S�u�=�`�fu���<��`���ܻETt�Ƥ,�$��=2�=�S���>GS�<p�<��=Pa�0��HD���4=�����Q����<�г<U��}�=I?=ZY���p�֋�=NR=��2�VÁ��];<De��{�=MT�=$�ûl��=4Nc�U��ڏ��G+�=���q'~=�b��v�EN��2oA���=��3��_ѽ/yɽg����ü��<�_^����`=�:>F����~������.�Pp��z
l<�-��h�Nu������M�r�;">5w<�E�����'<XN��mN3=�N<G@��wS��'����U��;H�,�u��u�������7=i�7<�J�=�b=]�����=ZM=]�Z<�Q�<%_K=λ��&���츻v��<Xe�]\�|�='�<��;)«=M���am�=A/=�����<�Wٽ'��"eh���:�TA�=z���9�=d�E=�	=ё�D�*�R�;���=��=\�=D<=r:�Yp=l/%=$ˉ<�;E�9<Mo=��ս�i��4��A�����<�x(=?�>��ʼ��ϼ�����H-��Z��\���m\ =��=�='��u�(�p���%�_=H�~=3�����_�r�׽�z��
�=�V�=jK̽��>�.����ɞ��1����^���������<
�=��>4r�=ެ�=nm?�ù�=��%�(�=�Lb<� �;�n�=�T(=�z=X�<D�<"\�~��=o;*����=Z���ֹ���=إ"=`�=�ъ<���;D�<E[=��<�9��_������f�����2I�=�|��ʌ�g��;A��3C	<w�����<�xI=�/d��k=}>��B=����~<z��zZ��~"�9	�/�;=h덽\{���<�ꚽ�{	=�%E=������X<}��bc��� >�1i=���<'Nu���]; ���e5)��yܽ��d<�R<?$�=�>�g�<Fm^=Nb�=�:
���\<��½ㅽ�+��q=�y=�ï��Xɽp�����'��}M=J�����;*�5����.0�<-G=�d�a=�o�������H\���=����o/���_=bYT=��<+�ܐ=�>w��=�ɓ������픽�r��^Ӌ=�7�;����v�C��K�k'$��-�9�`����V���}S=$���G;t}6<�yq<��<���=�,<tc&�4�R�U�w=����+��M=�^E��|����(;��=�]\�P�<H/���򃼷���í�<��j�4����=ݯ���$�=3�=E���R~*����%a=�ս�K�=J�==���蕽[��=��U�Miy������p�:iR=ȏ�<�Ŕ<W�T=��K�6��Y���d=���<k[�<"ϫ�[W��O何�q="��<�]?�Y~=5J-<}�=�l���S�0#��}�����u=�]��w�:.�0=6��=�j����<*�����K=Z/z�k>���x�k�P;�lW��A��%�κ7Z�<�6��=����9�*�~����<D��p�=�=^=DV�����<̉*=�^��8�=��=�x
<�P�-���6���I�;7����0�����-��V�<�`/�_�6��&�<�}ü-a�<|�����z+�:�;5=Oޔ=� >$v�=񷇼�y��� %=�ʇ=:�W=����G�Q꫽%���qo&={є�v��N�=��2=y��<ɑ�=�,��̝f=���a/-:�H�<ߊ!���<�/𼒩�_����k�����<~�b=_%�=�h����.���h����=i��;�"��ν�/-���*˼9l�<QL�˥$�݆3<��Q<���J*<>�=M�;̾b;�X7��l=�t�<@����8�VꝽ��:��;�i[=������m�.;'��<��B��no<s���~k;I�Ƚ`�H������=gs>� ���7���m�c�ݽ��Y�4���e<W2<�8==`B��щI�ă��}�=�Դ���@<�����P[;5�l=���<'�=C��<3G��S�b��=�y[��R�<r����;N�#��Ac=��w�U�弃+�<�!�= ��=M�����X���ف����=[Z=U��=��	=Z( >d)<��e�����< �=�2���^L���x�<��<�'�=mxO������V><r���O0:�ZVf=}S���i=�Aj=�yt=�$��fF�=1Nj�����+�ý�z��k��w�7<o|ּX�
=�k<���<�r��>�}��80�Z�"�Խ��<_��<!��\!���/<1����=�9��� ��Ȍ���漗���'�輧�6��֟=��l�M�������=�G���%�<������<�w�<�<�=.�2�g=0rʼ��v���~=�֋�	/����<��=�4��}N���=j�<�U�<?�=p�~�Cؼ�鞽�8<)ĥ<�3�������\�O���&�=�]\�ٮ��n�!"m=�H��4.|�:h'�n�$>��=��*>eX�ۀ=�T<�Qш��@����=ۖ=`�����<�����#��r1�����ƴC=�,�<J���������e�=U�����2�3��<�0+;�@�=� =�W��DqB=v+ ��{��"�=���<k����gz���<�O<�n��kG<x(>="0�<�
���#)�NB=yô=�b��i�!w�<�%�6�f��K><'��;/�6��p :&[���C����Z�0����2}�����v
>"Z=��R½�
���}�=��2�k�0��ዽ�$��1���Lp=H����	�l1!��/�=8��m�2=� �:ӴԼ|�%�κ��߆��}�l�JEl��u�:LP��m�ƽ\�g=l$6�<�໾�=��m��=�SE�vc+:?v�<yAu=�Vb��<>=�❽"�\=>��h����3�;�c��=�4<�մ�)��=�����< �T�1<cTv��㞽�y�Ȭ,��m��͙��4������=�P�=�š��Li�^;��
�;����{>B-��'����p�����=�0�$����9�&����6ƍ��W.���7�}��=&W6��a��r�A�=��;�v���3O���<��⽟I�iIA��=U��6B��L�v�G<P4��{�<��<畼�>����R=7�<�iU=604� ��qh=QMK��ㇽ��Z�-ᨼ�Ң��r�<�;"�S>lP˼��%<��.��<�N�dZ�;�$=�ӑ=<MZ=N�)��e��DU�N�:�3��=�KW��v��iѽ_�=h�<ڕ�<59=���=k�<�=-=/�N���(�k� -��Ú#��h�<�?��4�=o�q< y�<B�7=��=ĖA��� =��=��=.Q�<g�˼6#�<m伏�V�S\��/9N<_���ׅ��t���=(��=�n����Ľ��>=ۡ=���;z����<4����P�;Ӹ�<�˽���=��½���1!�@�:=���&���K=8*�=�C�=ea��꫼�ʼ=��c= �;�''�<r��;�q�=Wk����=�j�=?yŽ
�żI�$=<(���T�̞h=�F��{���:��;�NF=��*�1}�06���w���=�=U�ɽ>,ݑ=۱�<�KE��r0��E��:(
��.޽V�=hX9={#<�Ȧ=7"ؼB�!��d�:�U;歿<�Ϳ�0(i�I��;~�K�!��hō=�[���$'=��=/p3>���:��p�)����=VAO���8<!{��N����<ú���;W���<�C>0�|���P��鱽��q� ���
=��=�Q<ڙ�;l�
�1R�����=��2���#=pNh���!�O���9ѽ��J@�=q\���^�= ʻ<���=E�����<	@K����=@�;>�<����W����=Vi<=��W�u��<em���.>!p�;�����½���=��-�;{I=��i=�Ҙ=���=����==�ŽVm��A>�w��U���֗=մ>(o���V�)����<)U��������=>�$Y=�d�<���=��_=}Hƽv^��l�:�x�ܓ
�]���JBe�xһ�oN�;s����p�da�<��O�=�'���_�a�<wsj��X��od�<��D�H�����u�[��;f1ü����==�:H�;D�lص�ĈN�A����`��W�=[��<����!=T?��7?=y}o=Ϋʼ�{
���ԡ�<!����[o�#��=7B��x4��4>Tl8<*8转
�<��;T��Z�;�ڻt>�r�<ǃ̽��P�1�B^=�2��̸�<�t�<�S���¼4~�Ἇ<���=x<~���\�={��8�w�8 :�k)���q�<xt=���=�ϐ<�� ��E ��l���í���=5�8�,g�9S��=�>*d�m4�)�(�����S�=mݚ=Ҩ<���{A�o�k�*O��X�U>Qb
��$�������1=�AK���=9$1��e�=�EW<n��r>%,O�tZ5�.k<2����D��M= ��;`m0>��6>��P���ʽ�7���(�<�\��7	=�Z���q�`�=�Vy/={3�<��?<3"�<�I��"��L�o�����
m=)�,�d0�=,@�<rꀾ��i��$�=�G��>P{������Z=����j���"=m���=�9���h�<NǼ�d�=x��SP@>|����<�
;�eչ����Rs�ŋz;�_�<�7��Pi��x����߼�T�=	K���ƅ��|��X�k;��&�e�?��k�;W�c=JÝ<��]h����=��>4ǽ�MH���}<QHg=�J�<��k�߾�5E%<+m�=+�<��=t|��
;h��+�P�ka����@��u���d+��?�9��Q���=Mu�=�_����#��TG=�Bw<4oJ����=4��<Eq�=�Ƽ�p�1�ef`=��<�e�<4G<�i�=p�<S�W=�둽� ~����;���{&�=�Ɉ=�Ƃ��E���S3���;ĵ����]=��<*��=��=� ���V�u�<������4��+sԼ��=n����l��yw�|��=���;��d��$=��� �=1ϼ��b=$�Ͻ�']=̧��l圼*�=�ҽ�m�����=��̼{1S�ɥ=H���������<J��=�E=�)� ��v��=�a>մI�fX]���;���Cv)������=���,�-��e����=�J�1V��ا�����ј[=�x��a=������)=��<А= ���٭��0\=���c�����<�9=�R�=m�3�ˣ����=��<:��WuT�[�=g$m=r�=�J���Ľ3�;�˽-If<��������)�ڤ���u>cC�<�3ɽ5\�=�ٻ��e���=9��=���!:m�������C$�A�?=��c=K0>=Ʋm�������:�uO;���=х<1��=OQ�����@�Xd=�[	��=��#=e�_���.�Ž�%	��J?��ť�y��=v`�j>�=x=icŽH^Ǽlc�5 =�S��Z���1>�U='̋�c`��D�9���<��2������Y�����=Z�6�M���w�>�si���?�M����<�0�<PKZ�fd=P�_=@��<�^� ?�=؜ռ�m�<_�3<��ü�5ؼ �T��J�=��#����= z]=6�!=����<�'<e�𼀧�</��r'3=9������b����Ο�=z��[�)='���Zb=�5#>��y�܌�բ�=ٽ�*=��|=
�<="��~�=y);>6��<O���߬;�i�;{$=������֢�=�c�=�ux<�Њ<!��<�\=����]��,q����9���������������8���<Lݽ�>V;g*�=e��<��+=D��� �/=�A0=�"�<r�������\=����ﺽ B0�7J=(.J<\p�=ؤn�oH}��2�=YT<�@=��������B<l#���֒�G/'=�=��2�X��Z����<��)��GF�v�;�	>��L�jh6<w9=@~ ���@;���=�1n����=Zr�<0��=���Nja������<�B��|�-=��<D���o�D��p'=&�W=��=�:��P�=*�f=A�;�;��:�d����=�!=�ۍ��u.�D��hF�<�RĽ]��<�i���/��0j<d�@��� <��g<Z���;��g< \h<���o�0��Q��"�<k�i�:�q��������X=T&>{�S�<�9�"�.9=�;���V?<GI�;6&�D<%Sq=��>�ߧ<�32�m^�ߠ�����y��A��΋D���;\jV�`���_� >< ����<\R�����'[J<�)��������=����kP�����=V��=J<B�DL=��1<T�ս�y��� ��C�=��<�ۡ=�����<&�ƽ����#�N^8;p����q�<Y�-=R��;�N� r�1M�;��<[�=tU*���4:Iw;���x="]ٻ��� _>�T=��U=z[2�gs����};|[`:2;^������<.=TTp=wT@=3�=ڿ�vg:=R�r<)��<�Q���Pa<I����d=�l�p�!�]�!>��R�v�=贅�F&�<Ŷ{�x;;6s�=&D,:��=27��>�C<��ڼ� �X�4�wx��½����.��=�U)��|���<�;��l����Zb�.�V=Ҕ�=���<(�=��u;�@�<���A,��L���x�<P��=��e=zS��˼�B=�|�=�t=��8��<�� =B�z=�\üEG	�;G=�`=w4�.YQ<�i������`=�=�=�T5=~��<[�[<�P�p欻��S��C�ByʽE�*�Qm=,:R�X~�2_�=����l��Y�=S�=!��=���<�Qg�Ҋy��$<Ԉa����;U��=��̼&v$=4=#�"<�r��,�Y���S� ���-߼���=�yO=��M=�� =L꼽��5<����x;8��'<��`��㫽��<�<a�_�o=��<�1>=w�u��/� b=��^<�*���d˼�C��]>%�;��;�`u�=x�żE��=��d���L=ͱ�<�C<=F�i=��K�6�=U9���,N=�����<=�d�=���W��Q����v�C�<.i��-=�/�����OR�R�������>eS��:�>;������<A+�v	�<�%��<�=U�ɼ��½����uC@=p�O��v;��<�c��:O��(����m��=?�>�u|=��<�� ���/���=A�'���=�<�<��X=�h�=d���D����~<6���м��=�����_=j��=DP�;5�7�R�A�6<M����i�bk�P�n��h=����@�)����$݂=��ŽL�!���^�J8�=(Y=�10�$�;=��ʽP��=E W=�"��/=��q<kY�<�0�=���=5����I
���ּPѼ����r�����)
�=�<mt�:�c>�#��/��X���#�<r.�=�Z9��.�=,�>�=Ɋ�w�d���<��<���=�i<�=�<Z���_�1>P52���Ÿ=)+���/�=���=*��k�<;�&�=�}�;�f<��k<�R鼖^"��CW����=�\��|�<�MS=L]�=������m��A��<�r=���<�G����9=Uc���6<���=O_3�	EF�D��<T��;�<�<]c���2����;�v���(>=�	=��n�Ic�<���ㇺ������M���,��=�D"��}��š<�"ް=~�=D񖽡n=Gj¼�<R5}=�I�<��˻�W<�Vj=>VX<D⩽��r�HI��I��ӻ�=�)�Gׇ��L= K��{�+;Nx��69=t[���y�<x!���P����e�=&6����+��[��_�%=t��=_Vü�E;��\%y�-H�:�6�ԋ#�u�;g�Hj�H��=�F<��=��=z�;�[=�-4=R@�<��R=��q=%vs��6<ܠG={��=��=��=�8�<�}�<�ˆ�u:�����<@ĉ=� -��j���eU�x=�=��<!���X��=`�m<t���j�_�=Tv�=���<�DM=5j�<#]�����dAH=�n*�x�X�����V�L�����ν$g��2�����ϩ��(0�x,�=�?6<L�=�k:���%�a��m�<��=�`=����&�}�������|���)e�x�B<����ZD=��%=��F��:�������ͽ�K�ߔڻ�Tv�/H���]���Y��㼽�y�=P$!>�N>���=��<�軧�2����X>G�=*4�= տ�"��;���u)	<����8<<u�=� >�
�=E<j�~�<���*=��\8Q�ü��A�4ff<Ԣ<��>�z�;��5�x��f0<p�z��$�'���W�����<�,�=�� �vA*<�e�C/��c=���<Yc��,f��v���v�p=>yg;�{<��:�׽�����@i��ݠ�x8c=m>�=��O�=�|����@�=tF^<(t������Ƒ<�$���?=�'2=�Ql����=�P���{��5&��͜=<���9�����o<P�ڽZ]S�>���q��c.��Aփ�i�.=��b=;>t�_<$y��	<E��=V��<ڶ�=bU`<��ƽ���	�����2��J���~�=?.�=qjR� y>����������+��<e��%Y>�E�<3��~r���Z=��"�ʴ���ʼ0�F��wO<���λ �~�ؼo2ȼW����=�c��0r/�a�:�gI;�M�
�=�6?;56W=�5=rp����7=�	�>3���<6��=G�X=s�=�̗=��J�<\&�<o;<=@Y���=(8<��=�q<J'�<=�J�q�2=�����4���8%=nx=��=�ǽ�ow:�q���;�?��T ��R����]�>=�w[=�Y>(U�<�3<�D�=Z� =˸�)r�=����*/Ѽ@�O�y�<�U��vI�� s��h����$;�!���`d�
`#�[�+=U�
<�:�j����j|��M=m�|=�Ѽ��= ��=)��˳=C��=�� =g}+�M~�;'��XNI����<�����(<�΃=v'#��Wļf�/���>�>8��[����5��O���i�=?#d=a����#���ĻQ�Q��K=I��4�==��b=Ҫ޼�ѻO��e�����3��,��;� =:�<?/;J�[��\�;����_϶��7���;ԅb�Va'��2���r�缁����<q@o=H�2=c�=�"=Kѣ�D����h�=���)o_<����=��;eQ<���=}�����I=��A=�s�FH2��2�nGf�H�1<QӢ=B<=�����Ͻ8�<f�=��=,�ͼ��b��<%=	��=~`�=zּ�5�9��=��(������n�_И�q�S=)��J���������=�9>;���=��=���<���;dI=qwQ=$�=j�<����et��ڽ����<:Tn�=<��<=�z��㻽Ç��o�<���Q�����=AFh��*���������:�׽;ʮ�Pz�=��[K�Yi>�&��,��
F���v+<��1���&sh����=3� =܇���XG�y�<�t��Tĉ�����w�ּ81g=�%ƺ��O�Z�B=Bi�=�Gd=�G ��=��=������[v2�~Q%=�;�C���܄�CcD<�/Ӽ�ic<y�I�,�Q=��=��^�8��<\6ּn=A��=�o�=b1<hj�i�Z��=F=5ʽb*=�;�=��N�盢=�2˼�z(��7k�\
�I�=4��Y (�dg�<�DM=z��=T|��\��gM�<U�<'`2�����X��?���H�=�ZŻ˧)<�1f;�#���G<�t����<Ǖ�=6?a=C���KhR����<A��RP=2�<���Ȅ�<(AJ=���<&��=�9�<�K=:�)��=�*��Y4�-�:�L�L�慠;+]f��D�<Ctl�t��<����@1���'��k�:�#�|Zv=���=�x�=%z����ۑ�0$@�������=��2=$ӻor��d��
<���=�M�!�z��xX=���b*�=��c�h�9�n���� =��?�Wz�;u��&#�*�0���*=m+=�Y�;�R_=�*�=䱛��:=�d���>�� =�����#���(��+a��7D<Cإ��?=�6U:�S�<i>��A��Q���6�#�=;+�1*&=�xֽ�I��g;)=?��;�=#^9�{�b=��;�W˽�R>=,�>�X#�������;=Eq=�z켤�1�|H�(MW=Ȑûܿ���)ý<�n;��<�s�<p�7�u�&�_��E�=�M=��=a=�K�<���s��V�>VQ�=�)�=��<�<l�޼Y/��I�=�E�\&�����=���{��2�;=��=QRe�R>=kN½)½��<���;dk���M�<��=��ļ����w�=��o��=L��ν�/�;o��<Kܞ=jd!>��=���բӼJN=�>[[<��<�ۼʂ�x����(�@\����>��>�f���&���Ͻ�fŽ�jb=zH�;p͒=��=*�;Ђ=�O��8B<i��_P� 2�x �<MO%>۶�=*y�=�٥�ؽ�����)<�O�<D��<X�{<�L=r�=��<�̞���-<�9��yĽ.sh����׺|q;=Q���pa=�m�=����ou��h���닽�l:����q�(;_�z���꽚��=0 =&7ѻ��.���"���e�=C<#�2����<ȱE���!=I��y���+���->�D�������=�9O=���=�y�=��D=����볻=LEz����((���������z���P�J��T>�T=��<�� ��4=��Y=��<�`��Up<#���xd
=i/�<���;Ug�<�Ս��ه=k������<Q,�<|=���t��7;<.%e<�倽�l=��{����]�;��>���,�!�x=:?>ŢF>hJ�:Gڛ���<n	���3:+�<�<Z=��R<��<K��Ӡ�%~N�&2F>��ټS&��An��'�ļ]%��Ke�=󚅾Z9����=!��=dvh��"��y�A =o�㼎{j=����،�=�g輢:^=��ս�=
S�=r?=��<K�ڼ񰻽�[�=�q+>>ԗ�:��Z��=��1>���4�<�u���s�=ZŖ��t�<�/�=��=E	���=�"7���=�PF=�{��X#�=�,=�c4���-���;�G�=��:���<B5���q!�HD=�[;��:�%�ut=�4���z=��<�}m=����W}
�÷(�h಼�3=lW�;sV�<��<C���:�<j���_���w0=�о=0ݖ���;�]s�=%Ԋ<����z1��3=g�=�}@�Q�����=�Y=;IeI<ؽP��r*<�D��늼[g�<�Pb�͉;�\�=��Uò��P9=���"L�<�><�,��rg��Y�;L��)B������A=8k#>�ܜ=�ܓ<��K=�Aͽ/�=�v��w'�ex��d�]<�/B=Jz=�Շ=�A=`~½��}��!G=�`<Fu�=(�(=A���r\�=�R�=D�*��]��}*��d�$���Խe���_�9;=���=M�X<�~���	=�ު;�C=�7�<u�!���<�z���x>�=j@=�6�ӏq���6�Ȩ�=m��ܬ=���=ԯ!="2�<@X�Ӎ�=m��=�N=�U��B���^tB<Q,ɽ�½�ߓ<�	z=�(�zO���)�������4=¤ּC��=j�=!_i<���=�(����7>ν�K>m��=�z�<��ܻ�=� (��D>AҼ�c���4�=��<�p=�K����]�������W�<ѿ�<Iý���3>�u�;LȖ���<h|��q޼�D�=��-��h	=�F=�'缇썽���:�
�`�m����e5;c�=�6����<�5(<�v=����z}�<J��=:#ӽ��|;�o�=;��=���Y��<��3>b0>�f<y��sG�=$�<<��2�����?�����`f��k;�=�=�9�<�Q����P��%�)������L�=_�a=Q!�=��=�۽�-�������O<����e�[<z�H�M3b�_� =X����I�}n%=4%輛���d=�8%=Dd=T$߽���ũ=<�Ҽl�"�+�>%?�<���;f��=����)W=Gg=�&��o�=���L��8.�=`w��=.�;��Ш�<_���kw^����=J�%>�R�=E�u�́=Tr��9�=�P=@o<YT�=i�A�P*;�[2=�nC=��=�U��M�=�eo�C�Z<f�\<�>�<��<�����Y�<�P�=���Hૼ;�/��7��q$�wD���Z�=�4\=_��=a�}���<ϲ�;H��S��=ik�<M���O��i)�R�;;�,!=�`��V�=o,N=��j=��h�A���e�=�R�=��L<{u�ě=^of��ʑ��m�q��� �=]��=��������'>c��=5_;��*�(=�'��<��=��F��-�9,D*������76��&x�B��<�|��r	�:<��<g����=
u=��R<[ �<�z�=2x̼Z=G=犅��Ҋ����{�]� �E�>I�2=|��=78z�S�U=��\=l2�ݰ����<�n��[�s�����<�5��p8�õ������6C��X#��&��CZ��^�=R)�<Q�
>��=Q;�:~���W4=�
����!���Ľ@x<��="�w�Y��=Nzʼy=>�_����;`�9�w=��s�v��,Bc=H��=/�.��<� ������C�<�t����XB7=@�)=G��Ik=4-%;��K=ԘW�Ћ¼7d�<2�_�� ��=�G^���A�Cw>�-<��'�W�=���=q����{<�";&��J>+V�=!F�@�����_��*��7����*���d�����J<���=��<�4�zk�=��ɼ(���>
=7Kt����<�}�ü�
�=�<c�\&�=0%�;�`f=�h���;��ʦ�;hۤ=wm�+�ý���bν�,�=W9�e�=���y�w�Ğ��.t����F��=���<�]; ��:o��S����e=�>ϙ>���=#C���+<I�'����=��{=S��8 =���	��݈�<[�A=���=�Y�=rR>=��`�;�Ǽ���/Ϟ�/9^�*�=��=_ك�`�/�I�O�~�<7�m��M���M��9�e�S���!ẽ+h=V:������j ,=pϜ������r�<u��V�/=�!=F3��V�7=Pr�=EA��n�����-�������?�=}��=aW���1���ܻ�7g��}����<=�����=Z�<�r�=���=��=�;�=�˖�9F>�?=�k�=9C�=�C���#���%��fn�=��r��#>�b>��>��Ƽ�����ƽ�=�,Ѽ!b�=���==~,=F�<�a�W+�=&��a�=�I�f�$;�[�=�݈=�Qe�����I���_�=x�=O���AT>�W������1����J����k�=�5�=�1¼Ն�<�Q'���u<��<y��M|>��=�uݹ ��=�9>*���c�!=��r<��[�~i�<�N�p�= �c;/	�o==1�;�p �A�=�$�<7��4h���m�R�;-|G<ܐ�=�\����p�~Km�FE�;J�<f�>=5����s�<�CG=$D;��=L��JѼw7��:4�G��2 �~7����#��j�:���=E&��LO=燒<�(<��=;��h��<O�=���/;Y��IP��ʱ<���=��<��a=�����=�$=-ު���=b� �f�D�Ά<|�ļ��v�vݟ=�'<������� �<�V�=C���u��8_=x�<�Z�;�{ü��=�"�;vj8<h����x���I��O=�7����=��=�]���A>bb=�u=<n�=a�=59���.�<�G3<\������;��,��pͽ���<=o�<�3<�]��P:=�%+����:�dT=��o�	2�;f�P��ü(����᯽�M�;ޯ"��.�<{L=&�h�.�<+{=	�߽a�k=���N,=3��<��L=�޽N̛���$�B�<�6��^p��wT��<�P��z�;G�4�L! �τo=�WK<"S��2p; CW=��<9����ʼ~�/=M�ȼ<KE���<D�=���������"���f<2=&�!X7=�F��j=�Z=�`����]����<�<{]�[�;O����<���=P��=������<�f�W�<�r�< �U��RR=�X*=P�>��=VQ�<	�a<��ν���=+�=�X��2J=�LF=��F�=�M=�X@��ض;-�ӽv�>��^>�D�=�v�NR�<򣃼8�<���=7���4�<��l�/���6	�=o�ʼ�DX��^�0��=�1v���8��x����=R��fH�;�=�;U=(!�^:�a(=�8��DK=Ĭ�B*�))�=��=���=z6�;O8=�z<�+�<���8(C=�!�={�1����dI̻��s�w?�=���j�����1��&$==�=��ȼ�t���=�^%=�U=��	�E�=�ɲ�g6���c;�&�<�E���%��Y��J=i�	�]�ir���.��j]��D��=Z�}����<����W=��A�[�`Z����1=-��=�#�<gm+=��<2�>����&[������X8<໽�՚=��=zz�M���Bk�<�7ڽY��<4p�<ؕ�� ��&5>��޽h�'�-�/=����ٸ-��Q
�y�>�i�=�ُ��p���WH�%I��W�~=)J����;<y�!=��齹�bw;)�2=�I.��T����м����I>u=�N�=�~=�#>�����:$��Ӷ=��='��������?o=MY�;��<�%=�j�A�=���b��+�̽2�>��p2=�Î=К�=
$y=��k<� �;-�o<��S=3��>+=��;��Z=<>�;�=���\>�E]�=\k�;ب��.8G=�<�=G'��b3=	�=��?�� T�˟罛�'���﫽�0C��i�[��=`5W<>3�=r�ѽ4���<�IL=3HҼ�I������v�<�=W'<d��=���:"��=ֆм�#�<�~=?V��|G�Y@=�>I�W��^�����c<�8׼�]��g���@����=�)'��/=�Y��c�F] ��T�=߇�<^�L=4�F�ea=gW=Im�<�$�=ƙq=o�=�KR��7}��zA0=�j �Xe��K�=�)=s�=J7b�����_��U`���V������^�f?w�v�=^�_����]�<}��;J�~�s�V=�g�<,dڻL\�	,�=r��=��<�=��<f	=,y�<d��=[=k���R�=��J�z�q; z<���\���C<�H�c�==�%�1��='�<�����1;|�=�p?��wI�`�J�@�j/�ǹ4<[��=�Z�<_6��
1=�N	>���=ǯ=S^=$to<��k��E�^\��a#=���=#��=�<�T��!���<J��u┻�.�I)=�<��=�:<ȣ =T4��`�_�^<�P=�9g=B�<JU��>����W<jY�<��'<ʱ��� `�$�=�h=����`����=����T%=�c�<5Ă�{ww<j�=o�<��N��̄=u�O�$��i��|��<��:�B�����
��G=M���|"=�܍���\=c.��~�<$ʶ��Ȍ=�O�=� ��</'=I�=;�:'���-�{����҂�@�@;�k����p���<P�L=PN��<�;���q�=����]â=»\��g����<�g=�o�<����#&=&Ɣ��#=�
F=���=CN<F}�<W�s=�h�:��u��OA�ǅ��jH<v:�<X��b
ȼH�������鼠��=�~�=O�=F�#������ۈ���8�%����슛��3[�~��<]�=�0��7�=I��}��Ɍ��6l��s�ڽ��f= .v��*���=u =�8�=���$ʘ<\����.�o>=����%F	�� U=�p5=���~��=�锽�N��y&+���=�Q�K=��ֽX�>=���=�ov=N�=���<�@��s;/�=���;u*=��=B��<E`<�֯:��=Y҇���<��⽑=�ɣ=�q��%�����=`��=%%5�{N=��<ͧ<���<d-��œ��^�=i;`�8�5�z��<g�)>���=�Ğ�K,M;(N+=�P�\P9=u�=J9� �=
=�� >5�B�fᅽoߞ�2"A=�����>�&�.=m�=�.��`�R��H�=B�p�q��[�<O-=�-<V2h��ٽ<vх��ܼ�]2�\u@<�B$=ռ�<�f��馩���N�5����cI���l=�
)=����R P=S��=I�����H����V<�殽��.�Y]��8-�QnT�O����(	�ħν@9�<���p���%qC=��<1�;�r=��K��O=Ǩ=�`�;���V_;��&B<���<=eL=VL<�ȩf��L� ���\Q�[N��v�ռ�+=��3=��=�Y�=�o���-b�ژG=���=���<L2J=�E�; j��߳<�<=s&�=f��=S��C@ѽ� >� ����= �>�|��)�A���>��.�{K��E�k=�����&>��нx��<E�h={�`=@�m��= ���=1�j��0�=.��<��<��=*�=�E+>'�=ؠ>Y�w=+��;az�@%�����<�*�=�h����^f�6���g4Ƽ�L�o���A�<\��o��������=��<�+�<<����;����i</[L����<�[�=#�?����<�F<�<�䑽����9.�'�	=n�;��&W���=~=�%1;Н<��(���𻙔����7=�=�f��Irɽqf���Ώ�̢��E�����;���=-��<���=ab];��=S�<�f�<�*ɻc� W½p�q;=r�=]0�=~��<F�ؼ���:,*߽[ټ�i�=]��=�;�s�(=޲�=ͼႼnYw=w�ؽ`���6۽�;?���ټ��2<�uƻ�}��8�F>a-;=V�0:��==JVƽ,Ƽa�Q��K�������/k�*�ؼ�z=���;Ȥ<b�=:~=��;;�<�����e��A"�<�>�s���+�7���x�k=9�R�=������=ݹ#<��*���m=:f>�z��7=6��<z�	��q��������T<������;i<l���*ZX=�`����$꒽
��d�$�)}�<�;5=*�W= ������Y�z��h
�ؤ�����=�Z鼽�?<)�'�Qr�B"#=�������<풼�?�9	��w��=)]�=NT��S2
>r?�=�/��i�=	f=a4��ct��͢�py;<�7�=�*>�7���|��Z��hд����;.&<<�ʦ=m����� =��H�U��+��ڮ����=�f=��;J_�$��<��=���=(�w�u�">�<��dhk=�M�=7E�2���=�ޥ<��Z=�|�=<-�;��v���Ž��&��U;�(�<�&n���
=�d�=D	>Y��:�X�:(���ƼA�w����<�|O<��5��~-�������;ԹԼ�?<[ƈ���Ż�nP�djp�"Lɼ�Կ=��c��(�=�����t4��j=I.�=\9���bٻ?�ǽN�=a�=;/����!�.=g=?D�<􍅽����ܠ��5�ļ|$,=jc�x��M-L=q�=����O��<��=�j�;�Q��+�;��=� O=��=�v�;"ӽ��Z��xh��μ�K=Y�"�"���� |��M�=�~�<ty=��Q���&<�?�=���=� ;����<��[=w�<�L�<��=���=��V���=/^�=Aˎ=��sN���'�LvP�X,��E�=�I�����=�C9���H;"�������	�<�Cs�O�= ��=%Qǽ)���`P�<���;�rP<��������<���=��=W�=�Af=�_\<�O�D�1<�O>��D=����lե�&~��8��=im�<_f>=���<�� �e��=$�=Ԋ;�f=s��<	;�=�
u=��H=���;g�O�tI����}�}3:1�h=��b�'�#����%�R��bn�<�l�<�j�=�i��[=��=��H=�=����������2>�jݽ�;~ҝ=�h޽�=&>l=��L=��>e�T�닛��h>��ɀ�s0�r?���ټ�*�=	
.<s >�x躎��f�:V}=+�>��=�7u�ʧR�D{�;$���1�=�4�;�*!���=�=�=z�<>�>R�ɺ�q�;�3�=��{�9��<��=����Q2�ۂ��T�:�W%��������9e�=P白�@�`�=NP������=򴥽�'��۷!=;�"=d�h=t��]}��Q֡��5�<O�C%ѽ-[����=��\�/�=�9�M�P���GO�=�O"��#���	�;٧ػ�����=uv�;�E)��)�F�>�=�i��9>v}�<�%��ʽn��=�<�<l(a=4ھ�R�=���B�=����<���:�"��x-�=�!�<���<fA����=����O��T��=���߈���x����U;R �8vY=��]=.�Z�6���}l�=��=���#��<�j�xE*����<�>=X����t2���Ǽ��)�B��yC*>l�����h��c������'�SO޺�V��}�-;����3N����=6�K�!;�=X�=�$���Ѩ=���=k�:��=�k������<m��=�F�;�IԽ?�
��ͼ�<�<��r=V��=�W:�=���ٔ=|���m�n���s}=ʤ=�2!=�Ó��a���4=�v�<�I�=[�>;�=}��<n�o<�o;����C���(���,?�kP�=8$�=8h�����?����GC�_�;�H5.�9`Q�@1���
>QE�="�żާ'��5���b���
=V�M���=DN��	����=D">q7u��E6��$<��=u/ؽ�zJ����=z�����M�x�'�^�d��C:=c��=r3P�E$W��S���Cy�F2A=�>>3����#=�����=&)(�����1G���ڽ�4����*��Ͻwl=��=)�b���y��K=�����߽bG=z��9uP;�1�{{��1�c=�]
>ҏ�G��H�Ľ�;]�	iB���<�Z�=�G��H�y<5�V���=�'=%�L>�.��6P�" �<��Z<#�q=,+�==�=3����%�K9�;7�o�F�h�u�HX��&� �hض=h�$=�S��~غ������I/='�J��D<o�ӻ�ǵ�0-e�}��]���1����Bu=
�~�F�<)��=��~<9sH=�Sw�9�H��%-=*�=��?�0��R�<���=q�==��6���t� ������<�����<>�6�����ND�=0�:<�1ռx�����<F���J/;P}<�:,=���=_?�=7W_=ҋ�0dr��uZ<�2-=sH�>�>�SW�`n�=�`��A��Z�?<Nb���F����TB�=��N=qV��K��n�,�y��<��=h z�"��<N���W��v���������C9�<���=У�<�&Z����;Sn0<�≽�輊�>1��Q<R��=��<ϙ�=�jS=�u��+"ν*��{�>x����I��3Q�%/ǽ�L��sC���=�f(=8�V���5cy�	R���v=Tt5�߶N=���=�#��g'�<�Y�<��=�j�=���=y���������<N����=#��=\}<m	)=%!���\�"!��@�;�g��P�t= ����n=<�v=�/�%�8<�׺<��=@�����`����9�#8���Ͻbz�=��<[0{�g�=���=lq�X4��+>=K5ѽ��g�G���jU<��}-=���=�FU��~)��eT��� �@UX=Zt;"����K���ie���6<�==�1�=�3���ٺ=�B�<�'�=}r����/�_h�
��JC=p�˻���5࿻'�=_��֭��-��=3z�<ީ��ƃ���'�<{�
d�=�z=\�� ��;��=�^8��h���)<��8�񾞼D.��c�=��l�t�=ym��K'��z7>�Z=!��=ד�<!e�=�ON=L���5B<C���2:�9�1���_=�v���z<��@=��Խ�Y�;/�(Y=zeU��3����W(X=�n�=�Ŀ�Z�=��Z�������7=ŒQ=�>��<���nI�����`���uT�=��T��,p<e��Z5^>	���n*��!���W��&=�_���=�������센.Fн_p���=ڼk��3��;�M3�)F��B�<�@��o�f�į޽�]�=��h=d������i��m7���=���==���7�e� {�<�*B�#O��5t�=`��=x7���Ie=���=����0�@��3��=�5��aw�����=��'�����z*>����5=�T=8����=*�ͽ�L�=-��=�g�=n�=M6�<�7'��-7=�b	�Б ��=_-,=ʬ���_E=X�p��3�;�Ž>=�8��/���=���%3ؼ`��=����}{ռj�<<!ϑ=�(d�<z�<�h����7qW=�3���g-�h�����Y <{��aш�V�i�^"�=.j�=m�
�(��X��^1��$�����=G�=�'���Yk����=t�;/S���u��O��=��۽�#L�E}�=($�qkG=��
>�����=���<ұ=>��<N�_��1�^=5�	>F��=��A������5��J;=k��<H�>=' T�4sl��+X�꣘�^]�<��Ǽ�N5�.����V=z} �����Ѵ�<^b� �=.=�Y�n�f��[=����������������=�P>y��=�ʽ��|�A�w�A<�,�u���1d=� <X�ռ�UX���ƽIə�#$�=��$���RuR����fCk=q=�+b��W; �!I=O�x��)�>�G��fRR��f��E=�;�=ֺ�=�����͈=H���ٽ|�x��۬������U >FQ<�|���|��0�<ނ�qˊ=	bs��%
=�Լ��w=���;���Y�A=Ba��WY��zY�~����CF=-;�y6,�5��=�Y^<�!����<�C�<>�k�d��ި�<@Q��� =��S��-6��FU�v4a�箧<�	=<6:� �2>�J�9�#��X�=�]�vR��W��=��>�p�<s��=YĽD�)=E6X>V�ɽ�e8=�~h=L�=%�<`�K���z��}�=���w�����<5�߻������=0&=�����T��j����<����%�;�u�<���ޑ=!�<�W#��Sμ;E���E�<�<��N�<F�ӽ�y�<�/>%w�>�&�=�|����������k��U=���=�d�<)��⾷<(Y޻�-�c�<e<�+�=��X�o�=��<��=x���!p]�Rя=@�8�!��=&�g=�:������:��a/E>UT���R�=<{=��<K&��)Y�ޜ�4��8�w4�E��=\{=k굽�̵���<���<]��'_�� ���4{=bݫ=�u:���=M�������i�=�M�t�＠�������(=��M�/��=[l�� �<�,g�=0�� =��������1�;�A�=�$��T�<v�v<CXz�y��=N~�=��> S=��k�����꺊�F��=G�>�];�����1 =X�A�K�7=��I=l�=M�R=����_;�<�=�J@��$X=;u�=����|L�8N>��H��(�m"���PZ<�p�=�$���芽�}<=�M�\��;���=�Y�=�B[;�����r��ǎ=�ȓ; $�=�Rz������F�������=v�=>�s�9=���=�=��=w1�<��=�� =S�;�?�<,7A<�<R���VI=���=�m��2ҽ�!<�ս&=溟ZJ�ւ=43=i�#={�"�\��<Z�:���=1[�=s�s;s y�Vdx<�$�=MЮ�׀�=�@�=�Y�=3F=�|<Ot=S���ǽt�<���=d��=� �=�;��ߟ�����w������<>]ּEB<�C�����=)=��=)rػ����.�h�]��=����ר�<h���\�����p�_#�;���;�:��#j�< �r�R*�;��d���<Y�u�b*ɽI0���|���⇼J
��|O<"�۽�+z<@�P[�=���<�E��l=�g�=�>R��|��
ǉ�ؗ�=r�ϼn�A�i����fh�S�8>��)>���=;�{<�i��K.=��=b��]�<hf=���3�̼���=�#&����X��e���䟽�
�;LB8���a�q�=�7���=���;'h�[��:V�=���<�I3<{cv<C0�p9G�,h�k�������L��Y>��Y��j�;*�=l�=%Ѿ�ý9���������k=]`��;+]���=��<b�����~�X�,�7^ܽI�6�Bt]���=�W�<�cc=�i~���黑��=^��m`�<4����=lv��%���d=a����8<h�=o���f�ٻ�`����=>|=�>+��.�<=NԽ'�7�� h��9����4��\b=�˥�z���J���9���h����U=2T���_�qr��2G
EStatefulPartitionedCall/mnist/fc_9/conv1d/ExpandDims_1/ReadVariableOp�
EStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@*
dtype0*�
value�B�@*����Q��==��=�ֲ=.���(�����;�f�=/ ��xL������Z�=
I�=��=,Ω=�:��_����<o�����>�)>��y�@��=�-w���=T�q=O@���[_�� ��H�j<rU����ٽ���I��=�Y����==A�<I�g;����K�+�=;߼�F�GI�=�7��q�<9>`�<����=<�=P��<	\�G;,~�����=��(�>�D�=ryc=<>�<�U�9$��r �=�o�=᫆��H�;���=�.">-
�=g�Y=��>�=�<K���#=�]2>Ry�=;�,=
c�=�Ł=��=ˊ<��=�_==_�6�Ӯ4���E=N��=*�=�P�=e�>�˞=��<��P�3�3�_}�����=ё�A�(=R>�=t�=���=��=��b�<F =�S=كʼ�*=��o��(�=.�=��=b�W����=�r�<��f��/t=s܈=L�u=�tt�}�*=z��8��\=��=�>Wߎ;��<�j��+d���f�&�ݽ/����h�=�9�=Y�'���������]�V=�9�Yi�=A�ͽ�t���t����4=������=�r�=��=ON��}�=��;�k=\�ν�R�=���=��=2�3�g`���9�����=_c>�������<�ʿ��] <O��j�>����?�Q��)�=k{�Öp�!Ǐ�C�X���<y���E�=t"����=�&�;���90m=�"5��F>����m(�������>�=^��=��>�i�����2G
EStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp�
EStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:@@*
dtype0*��
value��B��@@*��iټx�<4������<�P����$=�����-��Tw'��=�`+��n�;1� 1���;��-="���Z[���>�N�|��C���K����S�SqF=D�;5|<䶟=+�@=�[�=7���=���4�="�B��P=ٔ����Ti
�M�=ecI<�æ=t��U�n��=�=��0��=��lȉ���h�:��,e��[JQ��w�<�n�;�	P�� h���ʽ�_:=�C=\	%�T�Y�ɸ��/M=Y^=�<=���=���<Zls=ݷ��1a[��=�+ҼY��%5�=�.��
�=�<b=�Uٻ��=�C=��	�]��<�o�<�W8�p��"�=�%�7𹽮i'=0!���l��Gv=�n=��P>�3�`���u�>����@Y�L���v�<��»�s:���=����˥�l��<��<iM,=�(����<8�6���u�M;ôa<�����SA�rI�<�'�=�^^<ٯC�x�=�&=�/"�{b�=��|�#c3="�� -=������6~�=���5q��)��T�<:�B���4=��HZv�ƫ=��<�P<]��=���<кн=���vܑ��L�<}s=`����1-��8��t��=EQ>9�;�(�� "�w�(�ӭ�H�<�{`<��I�1��s�=>#G��p2H��#�=B<�Q�����=�(B�q�$�ҞY=i�7�S͖��W�gD<#�=���<C귽3t�=ں��}��<� ���=��5=s�`���g��$˽�z>(%=�� ���h=�%:��O���=�u�;XVG=r�D�ի=2�b���@�2�*�{�e=Z�>1�(=����	�A;D��ܬr;�Y<��R�e�F��=�>2��l >��� il<�M=:P�<Ў2<�ý��j<0���ض�<So�<'�P�y�����A�<2m�<�ue���<B�ͽM�r�֠
=x" =��_ $�įa�rd=��Լ�)`=�Q��_=ؽ�=1f=�L�=�X�s��=ː�=d"����C$�p�L��cZ�D������;�x�N86��a=%�ý�P���ʇ������;:�SJ�<�C�Y��i�����t���5�<�[z����)���>	f�=�a����f��A���T��>��T���fh�P�ὐ�*��E�� �=e�8��; �>p�=-Go=9�7��Vu>�n�9�.���3��''=t�j���!�]�"�d�=3�=5�v��d>�����<ZHν�H�=��a:������8�����t>�R�݋`�w4�=�x�I�	=��K>�#���<��<Z�=��<��u��b�����=���=��=���b�5�=g"�<�*z�C���O�q3����=n|ռ!+�d��=��<�Q=1�d({�L�f�M�7=-�>�T�=��<N��=����X��J���^;��?��=�.�=X��<�I=�$,�1%P=lO�� c�u���QG�����м�"<��0��E<u�=PU=���=���<V�H;ʰJ��ܙ�d;�=_�f=Y慽#�������r=Qt�.S�L?k�mԍ<ξ>���I=��<;� �=��}��~���K � |� J&=ӹ��2�(�<[q�=Ǳ�x�~=U��=ƅ�����;�5�=���<��6�	�<����|��<&�>@=�=���<g��8���C=>��ļ\l���5�1�2���<=��B=�;\={KG<p`��a���.�E=�Q�p]̽Â��!v��=�e���zC<��=g�>C_=����+�;e����H@�q��(�� ǂ��Fo<E��<���%��=��E<�=֏�����=�.��������=SE���={2��, ��#;���WH��(E<����O=���_�t���;Ǆ���<�|�;5�=��h�gg
;�`��.䌼�!=I�׼؍u��e<���`_�ڻt�sd4�w�����ڼ�߼􆀽:�$=f ��� �<�[g��H�s�*G�=nGý�����	����=�*��=�.�L#Q�(��Ǭ�;��<��=��T����+�2>n���h������B;(�=�.G�������8���x9�g��5��]��(�u��<��=���1/t=��=�h�=Û�=P4W=�ӕ��~�<��=�+;	�/=y�d�X߼���<<�W=��۽��i<²	>��U�ն��c����H�*k�=m�=�K�<���	�6�qG=_	�t ���1��=��(=`�=�cw����'_d=�)<�e5��oG==a���{l� �ּ��}����;b�=�H���*���Y���VA���=��&���'�X��<L��<F0��1���=;���)o�.���Q�=�چ��r<c�{�-�=��{�����ߺ..�;�|�=�����=|�->vn���N��~�=�t<��<?�>�$�M=s��^��>�뽻jM���1=֎<?8E��[��}:��kp�]��m���o4=6�i=�J�=A�u�&#*�������=��u=�8L���s����<歇< 9�i�=���<��\<�b;���<�*�<p<�`��Ȩ=�,%���c =w,/��L\�P��=�盼�Q��8$�<s�Ǻ �����<�1��x/=z�L=�i���Tr=�����y�=�#��)~��:;�C=	g���~�}#�񵓽��u�J%�=.hG��=�Т=�=����"����Q=C�!�1���=9p������KE=#ǉ<�d=�[�<��<U���L����O=^��Byx=�G�6����ѽ���=������+��k<�=�5�<8�+==�x:�8
=�A��� �=��u�m'3����Q!���{��F�.<����<4[⼴r��ќ�<Y����1��rJ�_�<�{<��)=݈����<�N>=b�<�8�9���Ώ*��%�<Ot�<���ŻW+���� �m���CL�&O�<��<�Z\��=мNF=��$�K=\�*����<�ó;�1���r��t72=3A=�%�<ߤS=��<대�d=z�V<�C�<$ `��[{<�Ѡ=�9�=g�O��%�<�}=;�f����>�z�=��9��m��'��KL��+Ľ�
>��&�L��1��<�%�=|�ч�=�C=Q��G�`<9�8=�Q(�K�߽��K=OV]��#���kF;��<>�.=��t�BѺi`=S�=r<�<�|=��$�Jö�D/1=�I�[v=�w���"���=���:� ���I;�(�\M�<� =Lg���>��=L����2=x&]�E��~r�����RR�W=���L<WP�=9Sڽ��R������k�<ݿ;����[iļ]�C�| ���X�l���v@�<��,0���P�,|!>�">�v=,�o;wK'���/������NO��� ���ѽ������!>�G|�,(�Z��=����6�Y >�փ�����E�󼈉�=+a=^z&�����o>�=��T��bH>��<�a���I�1����%��<����;;�W�l>F��= ���^�]=�l�,=����=x�"=�:;��J=vE�=D18�:��e�=�C� ʱ=�n�<�9!�+�<���؀���I(���W�/[��8�,ل���)�"�b��W<0Ѕ���E<<Pi=EG	�:��<�Z�z/M=D�6=��=�=�7=�gy�7z����b�ȼ�?=�Ou�����l2<������=��T<�̽�&��%5��b]��/���y<T.2���<n#=R(¼i���[�\���uD&=�%��@ ��������=���<6,:<��	=H�Q=�u=�<~� ���=���=�}<��
S=D&d<�*�=(�9��JQ=I�K=�K8�6��=,<��I��<ܒ�=��<�h����=�6\;�j�<��Q=<z)m�8ܻ�4�=��:c��v���2����<��R;������=�n����<��<��9�T�2<�J�����7�=:M�X�}��;�{G<����©�<Hf"��oѽӋ�Wu&�e�,=�gĽ�9s=��2�hd|=�W=BN��#��=܋�0�<�{�=�>��u� =�8�=�YT���=q+<y�罺Ѯ<��0�C�h��6*=����Q���R3�m}R�b��=�_��3FB=]�o��xt=���5<����t�%Μ=�)���+{;k�%=i��=)���!@�=`���G�=���<r��a5�����薰=�lE�i�����=�Hʽ�x�R��E0<��۾�<�|=�!t=�Or=��'��}�d܅=�������<�W@<��;���K�=i֭�w,�=j��"��=�I�����P=`�Q<w��=�E?=�ǽon=��<K����ꎽ'L�;�13����g�I=}&`=Չp���/=C�`�2�/=�1�:�ݼ��<'(D���=ң��_(��/�=i�q<���:fC�͇�C.����=�s�� �U�G��;�sֽ�㊽�� �ƽ ��\<�-�T�H�V�)=�B����	�Y�-<[��#�=/�>=/�����%E�=�Y�<�h�<쌼�!C�]-�<��Ԕ� T�M�0�
�{$�=JUr����%����N׽C�)�b�b��a=P�>g��-�,>�U<N��<2B�=���=���=ck�U&�* =�h�;��<'W�`��=���<ш�;�ʤ��hQ�>A=��/� �T� ڼ;'�=��=�`2>r|w=B=�u��Ĕ�d=R<<R�)���L}���Ts<h#p>S3�[x%�ۖ8�]z ���:=�����1��g=��?�G%׽�'��������<cH=���;�������=�?����C=�*��"���:�<bvͼZ�ڻ��<��� Y�=�M>�*�����25�;v�#�ڋ��9.,�y�=x$�kR�=�p�<rn�=�SO����=YXA�a����=ʿ<��<��l�x��=ʻx=xT�l��<q��=B��=�X}=4�a����=|��p"==�С<��8�[�%<uc
���f;�&>+��;�H>��ߖ <�GS=Fc�'����=YIK=sj���@=��=��]=<�F��N���T=GK���G��e=N۽�:;<u��=�!��)�<����U X��Ի;B�Z=��{���Ӽ��I=/����%�Gx�=a1/��6ؽ��=I_i�{i=!=O��c�<M����՜���B=\��VO��/$=��?�C�=�Y��BwM�[K&��3Q='�;��*=��<
�D�<w>�]5� &�=�g���Y&�+/�=X
��c(��Y=,�;���W=��f=�»t*�BW�<_��&�0=�W�=s��<<U�=L���=����=3���i�<� @<u�=�ڙ=��=����\<�}=v� ��=#��<{�-��C�;�v�=�)6=�v[�s�=�X=�D���;�=��<���w'=y���b��*�<�����V=�^c��WW<�U/=H��<(���,y�����E&=�0J�=��]���=Qm���J��@�EZ/=~�?���=r~�=���i�`<��S<����-A��;��=�>�<���<Z��<M��7�=V�%=���=��!��F�;�^�=_�/��r�=OY�=��<ڝ=�B�=ik����=qx>�4�=�E�=;�=�<��t�B<��r=L�?�03>,��<��=�p=[(=u�>�d�ŷ>�9>��-=)�>�>��=�
^���Ҽ՛�=@T�<2{L=���<�~9��B�+V��7F<���f*����|�s������<<�l<ɪr���=��#=<t.=����0gG�/��=hF=Ǖ�e@���=�!���L���=�*�<�5<w�=��=k�(���S���,�Pv�����=˷�<�6{�d�<��;�H��8�<�Q~���<��烕�剄�����b�=�F=��;�}	<�_����<n�����=~k�d��<�=��=~�f=cW�1r�<!��=1�;����z��<�㹼�n޽}�A=xq�<�a����=�=��/��aM�<d����T��Ay"=ë�=,f�=Q㰼����V��(TD=��\*Z��=�<�нx�e����='��?����ȼ��F<h='��C�=-]��3�<6�=���=9Y��}��=K$�=��Y<��A���x��������Ӊ	=E�"=�&���y=�qG���;�Le��Kۺ#�=���:�e�=z��DP=W&h��=;fu<�<�h'_<МS�[D=�a<e▼���<�����H$>N����/	�i��<��<�2�{���=�R�==)�<c6=d��<�>$=^�=��J="�@<#���*=u�6=5� =��&���=|� =Uq="�^<f�(��4�����J<V=�8��Icj�����F}=~Lݺ\<��3 >>����r<���9���Ⓞ�>	S=F�=�{2��x�<��/<��K=v�=@��=�!d="CX=V����/$<� ���X<�=�?��s;F;sz0����<�N<|��=��T�@y�h�F��@�=X�;�F5;NMʻt;�z<��=v��:z�B���t<��:����Dpv�@'�<y�o<��y<��X�
>�4=�/�=�kM= �;=��T=����fл �b;v@��]�0=ǽ,=��;?a>g/�<���,�=i� >������
=�>�#>�����)�=��l>G�[��L=���<1j�=�$ͼ�?�=��O>�y�\|r��=�;=�2�=2�=Q俽E���,����.�q�=0&=���������"�"��=��=*��9� �������A:��U>j[�=1޽�L�=Űû��=���=� R=��sO�<M	�:r<[��<��߼!���<�=Ш���=�"<=j�=�_*���<��=�W��x�=�	.<����M;f;ʅ�;��b	�x��<`�4��+����y����I�h�I=J����=:?N=��\<�������N�=_R���X-=m��=���<Z2>w����U��>�ڽ(1���ۑ��=�}���@f��l�<s�4�]|�G��=���-�_��nλ�+���>:��F�<*�=`cl=���<@J(���=:�|Z=j���m����=�Yi;���J�t��;Ј�������FG#�<�U=�bD<�b�T� =����8	+=]�>�o��z����\���9po��J�A�:~�=M�½��7�ܾo=��ܽ'2= J�=VV�<�I\<p_ =u��΀?�^J
>9t�9�Ľ�G�U*<�a��(���oY߼�T=�^J=�4��mt���-'=�!=�
�=>a<�6���˥�(ϰ�>��ko��?�=nX=M��p@��*�=ռ��&='��=��p=��+���Y�>�/��GG=��<^�=~�����\�<b/ȼ~�=���р*�H+!=�gҼ�i��o,���:z<�6ҽx7׽��B�˯=�v��]w=�T�d���T�kgػ�H<o|�>�=�ss�1�ڻ�(=�	���@�H{�TK��4��#� >��2����=�2���6�G�b53=�����Q=ˎO�����>�(#="㔾||�;f-�=^W�=lՄ���4<P���fo=��F���b����/���@�<�F$<#�=�U���⭼�l�;b��������=�s={*���G�fʔ=�H=�t�Yge<:�~�'=��)e=��;����h��dc�	�=�`��kV9��W=�nq=����]�<��<�\%=R�<^��_V;�ƼʼR=�T��UA<5�0���=�ȣ����<����({�l9�<�|����g�#���1���;r�<@&�=��Q=����^j��s=7Ľ�UŽSM=-�F�唉��#V�v���B=[� �{�=�Ue�O>�M���ei�j�<'IN=���=���;ł�:��˼�޼��X�����<�M�<��˽c�k?�<�%B=`i<�7Q�U@�=~pS=����3]���ڼ�aT=��W��I�<���{�=<�3=��=>��=�ML=L�=e�=)��;P5;��<2�1�V=��ż�FI�c�<�"5�p+h�Y�=]�=�����;S>�p=�wR��^V"<�A�=�j=�q�����x����<��=������<�ʫ����By��]�<�1��}�F�F>ig=�_��+���;�i���i�vQ�<�ٽ���E�=��=�)��_�<���=��=j���8�~�GZ=���꺼/�
�)�/�N�>���=m�o�^f"��>C�"��;�~��f���zNg��C���:��;k�L<���yaŽ�f�c<y�H�>i�z'�-�=9�߻���<Rj��6X(=���=X,�<���<)2�<��j�ϟi<
��<�͒<�L<+���M�:�M�;i�ν�u9�9�9;k�I=>�;��=��=X(����ؤ��Q����<��=)��<x�/<W\�;87�=;6y���,='���L��&�=S��b�=�&S=|	=#h�;��<=�+�=P�=w"z=m!x<M�]=T���K9ټ�i������߹#͓<+�%;����F��"8=^��<����:�?���k���F=�����iYE<��=%;�<3
��C"�<���V��<�B�<O=��\�>6���39�=������={c�.x��J�n�M?�<�>�=�
����>��<lkO=q)��oӼ.P��J�սֿ�������8T�=��=!��<n�\=V~)�`�>�"����g&C�|н��=���:5�<R�Q�-w��Ю�� s=|c�=��z=�$�;��R��t<�Y;P������Q���~�=��<�Ϸ����D=l<�=����Z<.F�<�j=�Ž@%�]܂=�!\�0fӼ?�d�s���"ż#;�;�(#=�o-�r0�*���K�wN�=���3�6G2�4RD=%����2]���<���;v�<?=�����N��=��h�Q�мJq_���P���__�}���������g{�������< [���Ľ������<�<�rr�;e=8+�<��<�Z�=�x���<=6�<��	�O+n<9�J; ��<�E輶����~=ߑ��s��l�m=�?<���;��=�޼���^½�%��ۼSw=�;�E�=�V�;�g=�c��+2�;�<1μ���=˗�=[�W���t;c��=1'Ӽ/��<�ڡ��߽�SY���=�8h���ᘽh�<9$Q�`�E=�)�=k����<9��;�����<0<<qn<G�=@Bۻ���9r�=M�ͽ6xS>����<L<=��Ԣ5��nh=	i�6]��n4��c���%9���?�u�;2��.�<=�-=��<SnȻO�����;o>��*=���;}��b�<n���A�=`���D�����o��=ʲY����<��F=}1}�y�
=��=����r�����;(q�>���� C='�K�/�v�μ� +>W��E�<4=]c��o�<�v�;ɒ����<#��;E��;M����t�=�9�<��=;ʽauϽ@����>��t���ƻ�ט���U�r�:ݏ��X����=�3h�~���]�=�=QQ�#k���a^=ڙؼW3�=^�ʽ��.��>�4=V���.f�=���H���
�==�����2��B�����= ��k0�<?g�U���+�=$y�=�C�W�w�*�<�R��g8���/:E����4�u،��O=�a��>��I�f1=��<T=�<<�[�����3=�+��������g=��5=�j�=>r��>c:�Kb�� >	p9�< <4����)�$��Pƽ�
l��!�<�r��X�/j=.����:H����<�_�<Um�<����(���<�䙷=D�'�#F���=��=6����=v�����=���1>�>�H��n�=|�Z�{�>�8�uh��^�?8�멽��6=-����b=�;��s��F�1����<|�ֽ�>�ā;G��<"�<EY����Ͻ�	C�繢="'�����ZI=�ۋ<h�;}��V!�ި\�ux�=s4H�y�
��u���9)������m�<�ݟ���x=2��ֽ��=�v�q'��Wk�Fht=�S=tf=�=�û�>h=`������>��4S��!�Ї= �<oa>S���Gr2>�#��=�ϼ�K<vb)=�mE<&��@L�pB���ޢƽ�t��>ݽ�;���*�=��R�N���x=W�ü�}>Ũ8;��k�l�+�~YE��q"=R�>�Ϩ=`�=�A%�<����R۽�l<��>b�����=�J���H��Ƕ=dS$�E��%�|��t9�U�5�OX!�6�4�qYT��d�=>u�<<=�k<�Y_����=�t��% =�覽 �<<$��-\k���=�H�����<i�%=��r=��r��%�٦�����<:��:���<����.޼����)�=�F�����Xٽ�I�\v�=��s=|���8&�<�!��2�=t��=FJ����<:���=>�=>����=ΐ@=^�y�-}弍S����I��p>oÚ=g�=B<���U��<e<�;��5<���;� _�ۚC�9_�dr=�����t�d���gy=��I=��X=QMP����=<�ؽ7q��=�O<F5�v��=���<�+5���i���;�_��T��;ݶ�x0=Cb�|h;��%�3�<'#R��엻D��<M�e�܏��i���-���?�뢯��F �:�:�7�=Y6"��a��=�M;�M�;�Z�=�Y��c~ϼ�y=!��=���s�P<H&,<�^�=)�ƽct�=r�=?��<"M��5�'=�:��6xJ=��=�ϲ��3����?=;�M<�c<��2����;�\�|~.�a��� ��\=sJ:=fJ%>뻽(;�&�;E�<#����=p�g�w��:p��[�)��F;mz7���;�I�>6�<�� ���R>2� =��=���=V��>P䀽�p޽:������<=�4�=zP�=F\B;?�e����ޓ�=#oE>ԣ%�}NŽ!^��q�w=1>��>�㸽h��<vY��S��=��9=�!�<a���Y�+�g6H�M��>�����xw�) �<&�D��� �(>�<(4Ľ['@>�m���#�;�༭�-�%<Ƚd��=�w�%�u�혻L��=}�;���_;�'	��b��y�=��$�H�fQp�� ��`�f�̡�5����l/��_G=�|=�M� �X=D��;	��qyK���B9fQ^�>T,=�W�=�8���<���=\4�;3==`�ý�ֽӟ��BH�=�0�;-�3��ݒ=������=�Q5��Zȼ�=�$�2� <���= ���qC�i�<���om�<�V�;�X��#H�<t[
>���е=
��=��۽	�&���������&&�<И�<7�#��g�����O罛L��jW�7��<	9��\�Ƚ���u��<洁�)k���߽B>A��[>8/=�P;8k
=ﳈ�z��i������,����<�H�<)͋�'��=9���~>!X�=�9����ʣZ=ٕ��;�;��ڼ�Ζ=W��\T>��N=`D0>* =������>q&�OƼYB��;�<�M�N����
b<����M>꤉=�07���=�I]=��D>&bX;��==e���-=��S��Hڽ3��=�ݱ�4�=n�4=�>������<��?>�q��r�7�ж.=A����~:����=3){=�c>�kݼ����~>�m�=�@�=Y�>�=[+�=r4�� �`��v�=5�)=�����	F�yJּV�=?�=V�0=���@y�=1�ڽ:�p=�P�<S��"�
�3�:�F/����=�Ij:N[�B�<u�.=�G'>�B����L<Q&(�e?�<C����-�=vV㼵Ӑ=��;��=߀	�J�A=}q'���5��΄<����������=�<ۣ�mWּ#�=�5B=�Ȋ�1�=�x���pp��p�=A�>���_�K=۽�����ռ�#�=�a=|0�	)0=��Y<�M^��7�<(�B=V3!=7�ܽa�<!Y=v��=��ٻ�P=QU��%�����= a����<��!�;�_�h��?>�o���{�_3�=�;=�^�=?�=�\U=�#����=Y����U�<�U�(�=:M=�Ǣ��k=F����Q��u��<Ỻ��;ˍ��t���̽�gʺ#�;��>n�u<�S=Y2=����#h��[z<��=9G�=��8;����+!k=m�ǼV�\O�<�E���.<PUn�����$=�-�����?~�����<b�(���Q��y����oQ���k^<��=u�X�g��Y�<�����֯:JLT�-i����<���=�4�<;~�ÿ�=X�<�p=K����^�\��T1"<n�9X�@�v�V�:�ʽ���������=9�"��=i�м��������=6�ݽq\�=&�;�HP=���<]1�=��u���6��S+���p=�8,�NFF�rTֺ�,/>k�<͉�e":<
������=�d~�z=M=�I>�D��8
!=F��r�7��	��=/<�<����=��R:�=�O<�^�ZD��KS���;ʌ>���=	Y��9�=�'�<)�=WM�=AQ�h�=�R�=��0=m=�=
շ���M���;�A��=5������&�=Ze�=/�o���)G<;�����=��=r���~m;֑�������q��" >"�+�_̍�%�=#$<�&�����=��Ի��[���<�r���ھ�ӓe�%u�<<Ȥ�ǆ�%��<,
�=I�=�ݯ��z�"��<��=���N=珢<�m��ק׽�8�w���(�=x�����ݼ4�=�p+�A��Z>O�<kn=^�<!d��wO=��˼�����:�~�� �,�Z�=�1D�H.c�����C8���D�t��Z[�<�5C��G�f���DEٻ�9��p�K;2M9==�=T=�=Ȅ����P=0�I�^!��/W=K��:9$<�D<AJa=�0{��%<�m�<��Ի�!}���=�_^��($�����㯽�л�=�輼�i鼣�;��S;�*����<aD<��U��y=BJ����Ǟ��=վs����=����,�=ݖɻx]<��<�V=�z�<��l��<�<���=�LN��W�; q(<��ż��<j=�;��=g/�-�MH��/[��GlE=ǘ�<�M����y�0�ͼ����܀��>�#������Gڼ��=EI�m.�<�0�r��=F�L9�	��u�`<!�a��.�=����0��jg=|<�-,�����F-;�����4�rH�n�*=LJq��m��d_=V%�~���(���������n�=�Bj���/�'>2�=`��<���<0�6=�=�
=ȣH�o�?=1�,�er�<�{W�AvW�%�T�4����<�#��C�%>��<�O�ϻH��<�Z�l��c>A
�Og��`�B�=/#�c�T>'��=�f���x=�	>6!���<�6�� ����|�}�]˔��hX�vx����e�1��=��齌o�=�U=���=�����=�������˓E=R�<�Q�<vlh�R[=�a{>,)>j����=��L��(�F�����g��u��N��<W
�p>!+ü����`hd���'�1߽�z��(����\�TF:�=�g&��sŽ�R(�ˬܼYH��sL���7�;񟽦��z������r$=DU�`��f�7Ob=�	h=��w=�8��	x��V�W��	�w�޽vJ,��D�[�=�/�ʄ=���<��<���=���ڋ�7�Y=�;�;3�<�����l��-'=�;��
�Ț>�MP<�����>����?���f��e��J�&T�� <sB�i��=˃>�ڴk���~��Et�+$b����=ͤI������������&U�Ȇͽ~o=U�꼻l�=�4��G���K��8��N���Q���l=�j�4�h�rnֽC��=�=�=�ŀ8� Q;	�ڽ����}�p���
�w]}�50=��l����Q�>ހ������S1=�xv�q5���ݩ<$�}���O_<�
�<|�i����+�=B>���x��<�T9���;t o;�e���U��cw��YT=Z�;`�#>�@��٩�����xJ=^;��EL���"�*M<��=e+�;O�f�ܶ�h��=ى'=`��;������=�D�Yć���˽����ɒ=(��jؽ�Q<&��=��e=����E��g������!���<�%���ɽ����=�>����>d�=*5g=�P$��r��!c>D�����y�>�F��KJ=,=��꼁Q���]>J�=������q>���O��<�̽���'�<���9����R�d>���<?l<����p��<�h�=�!��d�D��<KoẺ؈=�6�<�׼��=������:6(4���=ONO��e:<�֫;�$��=G��<ʶ8=֘m���ܽ���<@0��R��)���e=�_�=#��_��<���<�6���^�:�ﻱ^�5�	�,F5<��=��+��<b=��D4�W��>�л��k<!�A<E*�=�G�=��(��m�:j�ۻ��3�Z����;E�p\<�=���~�x!�;TY��]�<I\��"=�=Ǟ�<&�O�:;���ޖ:���=�2��T��B�A=����WP;/�<���%Z��J��=_뷻�Q��-7��!��8����8�=tᓼR*P�`v���P<o\>�UļC˽�g��[c�Q-=M�����=�j�=_M<u�S�]��6&=�M����=�"ݽ���:L<S�=�<av���=�0�<��<���s����=�D���9�{��u�=Q�P��g8<|���	9c�=*�8���)�"½aN$=���|4��8Z��6���G=�H���M~���>?E��0��h{���np>H�W�<*,�������w�/9U�|�(>�Խ�� ��l�>|�1>Y%��\���7S�|�^����R5��fz�<}�"�p(�����f��>3�/���=���<��E�wǂ�0I_>�����N�!#��ҕ=�`��	D���m�:�>�G{=����%��>��P�����ٜ���s�6^7=E����{=y<�!э>��x<݂������ϴ=\�i�DDa�v�<l|��H��{T=+.Y�`���5>ټYR�=�7⼉�ҽ��ؽN�=ŐB��xҼ���=ۑQ�N���k��=�����<��=T��=��U���ϕ<^�H:Z�i��Jr�N��=KgW��=��2=�r��{j���>Y�M�S������<�ݳ�gJ=�*�������(D=�^��w��[�>In���u�˼.j<B����=�q�@\�گ�=g.Z=X;���Ύ�G�:��}�4GܻxΨ��.컡ߚ�Su-=
�N��落}/<'K�=��<AT=}'G;�'���uƽ| �C^ݼ���<��������Xގ�YWI>)0=b`�����(��̴�����rۼ�ҽ�
=��=B�j��S$>�՟="r�=MU[���)��Q��`c>�!��J=��������r��� �=��<�J �l�;Kv>N��<�޽vlS�uy��vX�1,��/m�[��u^#>��a��²��*���8���V�p'0=�2h;bL= 04<B>�̺�ST
>j@�<�î=N�!>�p=YRT�k`=�v���u>������<�<������d�μ�8n�̩�=��F=��|=s�=4�,���m=
kҼG��=����P�E=�6��,��I�=�5��-a�:�S�<�>���=V��;T�%�a�+�ԇ=}�#���=���Α=���|��=�=�Q»[�?��m�=��ͺ�����2<��p���=	��,63�"�������ACV=�vžaՂ���!����"�����=>�����(�f�=?���g=˄�4#P=������A�l�����=�l�g�G=s2;���#�x4>�"y=�%����N<�<P����"o�R;����������C��h��f�p>�fU��߻��=��ͽ�������>����-1ý\e+<OX;p�����<��E7�=����S9��O0>Cnؽ��=M���B=�(r��1�,��|.=���ԇ`>%V�6s��@>ֽ�����<ykL�у��Α}�<�=��=�\������c=�Ɋ�!^V��P�����<r�=������Y��<)���� ��bR�:a��<��<�O/=J2�;/Vϻv�/��@1=*S���Z=�;����=M�ͻ~MD=��6�<���<�L$=�M�<f[���~�/�<�<�>��E���f��/�=�~�,k<b�w=vzV�)�����m�� @�k��<�t-=��<�L�=kU=`=�5˼�����=�}��6���N�=�=d�H=����ކ	;��T�1�
���<�
��՛ۼ'U���˴���=��=�c;=��=q'��h�(Ԋ�N�=��<��ʻ�����x��W�I����Mw;$[#=����S�\�r,�<-\5��5�S�u��x<V;m��K ��0E=j���ی�Z��=�[=�:	>2u�;Fky=�޿���<F����j��!=�Խ�=O=Z���D�=���b��<3�=��F�<�i�=rW�����<�BZ�V�<U#l�[�Pe�<�]��ж3�TX�<�_��̤�3�<.����@3=w7�;�:f;�C�w��<�=�>�<<���m�=(�<��׽��<����m1� ��<�G�s1=���;Ay��=h�)���=9��4╽t�L�� v=@ڔ;ޕ�;=D{��u�bv=C�����<��G��H����V<��"=������r��!��ٛԼ�v;��'�o����=bqg=?b=�V<��	�<C�ּ�_�=@�<��6��^�=a�i����=
^�g�ͼ��=;�e��8�=?CU�1R���p=�N��O��<ɽ�=�[����=�$�=��%=$��J��SZb��=��>���<:���[4=���=T��=�=��=�ݿ=���b��=K����N=T�=Ή<=v�ۼ�����H���˻��;��==V�=!4�2ڻ�*p=ݔA�w�� ���F���q=�^�=�d�=7�����;�<wYK= ��=bQ=V�c=�꡽1��<Fo�=�+�<0�<D���u���f�6��梽d��=
F'��
���^���V=�==��&=�.ί�=�ǁ=�(��y8�t�=x�]�֥��{�~=���=���ay�=5�����O=eV��;�)=ЦҼl�T<�Dg�� �NQ$����=
w���5����j;Oi׼���H�Y{ܽ�D��)`��VH����;0��:�S����լN��#2����=^�a=J�������sݽ�'�=禼��w=�����&�%o�z�
=��L=gA/;�Ȳ=䤷�ݍ�= �=D�>�IW���M=�9>���cμ܍�=}��=�sA<l�=�V�<�0�S�]Q��H�=J��;,}�������鼯�=��7�)yV=�ߏ<ZNֺV��=M����ß�ŷ"="�=z��J�=<�=��=t4M=�F>�����'=��!��M_��ժ<]�=��8F0�<����5R��E�����O@>�&u��<+ռ5(��᷄=ǰ�;Fe<	���K�?=OEH=�й<֗�:��}<�R0=��^���Rr�F��Mݔ���=�=~�������+>>C<5�=�=W�޼��r�����=���NaM=����'��1S�=�6'>�
.=2���pu=v�$=������=�bn�Ҽ��X{6�˙=�Wz�?��������Z⼏@B=�t����X=�/+�4���x�ڽ�m�
���=���<oT/��B=2ֻd�=��=���<E��<}O�?ɏ<V�j��:j=�6\=��f=��<7>I�<.h�j�k�i�����<>u5<�}�+��<�i���/<��h����=K�|���=�z漟�߼�~=D���H�;۔��33=���=Uc7�D ?;�('<��;r���,�=�<+L��������9�y��<��<j	�;Z%;���;�L=��=�{�=ȓ��0��;�r�=�]Z�rݶ���μu@�� 뼸zq<"%=�L��Ŋ�=�(�=K��=~>�>�<�1���o���N�fĽuY�fӇ=ﭤ=������=���=�)T=��`=ܣ^<��<��<	��:��=�(
���z=J%=UK=�m�=4�)>�X3��/��㳽~Nw��|=ʭ�<=�SO��o<�c��{�=�U�=�7Z=�cG��U�;M��=���:�⪼ȑ���<Y;�=��^�>�{�uߒ<[}�#��=r�n�P�<������Ǽ�3f��Dٽ�׼�����;�ؘ���Ȼg,���ՙ=�����<�ʼ=���۴�'�׽Т@�0��=n�=o�j���=O��=S0>�I�=�t��<Te:g/��E9~=��˽1DI=Y�?=�3l�}HT�W�!<����>��r���=�T-=���?|�=Q.�<@Eo�}e=�s�=�1=����-_=�����,�"�u�G��<uj»���3��<��<�RO�|��ȳ�M�)=$S�<������S���!~B<�$���{8����U	���`;ɀϼnu�Գt�a�
=�W=1Û=���<�7���F�<x����ܽK�Ȼ�창M/��A�<y��܇�<�-�����>��=���<�I�<X;I$� ��Ä{=S�L;1r-�Υ�����h=4e��ϒ�m�p��|��T���=�П��a��N��$^<��<x^#��E�=�R=�\�=U���q�=<�<P,i=?I�<b?���N=��Ľ����ڷV��ٻ�.#��<�=jiN<n#m�Mz�;��=>�-=�*O;6
��<��<�!=|a�A�ļ�a�=��w=Ci�=%�>��F�щ��o�<d�!+�=\ &� ��<�����ɻ6|w���=(�<�ػ�_�<�6�5��=�.:7���m٥=���%ұ<����:�<6�V�]{P�\$��,��<\%1=f/�B�ý��<���<��&�k�4=�А=ߡ�<�g_������t<���<�<4�������h�[1�������=EP
>U�<��<ϵt=�w=?���^=��=��<�����><�L=Gf2��X='�y��J='a�O��=�wg<l�9��`=5h=�ϲ<DQ�<�.?�����Յ�=����_^=��=��=&��R*@��Q�=�.��%��w�l���/���t��<��� |ٽ[F=�i!��_�<e�=e�<5��=������_�������L���bB�=p.s�
0����ܽV8ѼO�}����<�@�<�_��*�PӼ���=��9��n���i�<��<IL�;<!���<�=��T=̅�<� ��n�=��;������p=�S8= ��<oǼ��5����<���<p����,<����A�����k=���=|H�==H'�`e���!>�W���L#<�>鿴�z�!;��һ�^H>��� ��;@�;�*��@=�1<����ɋ�<E=C[˽$!�t&6>��{��N�>2�I�/<��=��=L��:5z��M���4뽒cp<T+���/��`�9:�#�U��.U=�抽���(�g:����q��=�rn=��m��U�/a<�(����>�;ԉ��o�-�"2<����CWy=���)�;b���C�=N�>Bo�<9.=J"e=���hX���=X7�=ӫ���ҋ<����_v�DS�<�1�;�"/���_�@Q�;H��<y��<�h<�v>~�����9=�T��������jg=~�K��H���r0� `�<_?'��( =���=��=����\��9?�!�Ҽ�ߏ�'{)<�v��w���˓=�q<�>	k�;��0���=ʦ\����=���N&�wk<6x����H��R�<҃q�4+!��A���J�<�W�=)��=�<�i��<�����.�V)��oe,�;>��Pv��]>���=�iN<B{�<.]�<ފ=C��5b�<!�K=�j[=k������R��;s�=*��=x瀽���;�v�;�|-:x%�<I>�<�#=`N��
��-�<�ww�ϰ˽�R�S�=�UO�<)i�( �=_�޻��=����L�7>�u���<�o<��<��;6�; ��:/>~�;b�<�&�<�U=p��7H=��L�pqȼ�����d�=�
	=t0����=�D�=���=��F<\�=���=�k����G�ZI���&=�q��ʇ=�,K�S���	=}D��BXZ=��b�^�<�J�2^V�+#�=j�,='>y	���2J=�8�T=�����gz�N����᨜=��$=x��l<���=�w>��_��8�<��q���!���C<����t�;Vڢ�;��=�kl��Ë:�=ʧz;?y�<a�(=W�<��<�<����">����sT=֣=�	=E�?=�M=�O���e!���.=M� =ޫ�<ͽ�=��=\q=No��lIG�<d~=Q�w��=����_z�t|m=����Ѿ=a�+�M�g=�[q<�����ùH򡽇_Ӽ�1�< m�=�����!�	�<�{G<�H�����<���<m�<�8ӧ��f^���b�+������P6y�<L���e�g=�9佼)�=��5��OI=%@�=����ׂ=O������:��r��4���=��O<$�6��� ��6м��1�g�l�8 >��t�z�<Rߞ���g������1>��L�'���.���">F��=�$>�)>�����J�<���=�ZŽ����vvN���0�6��=B���7͞�< �Z<S<�����*>�=���UP<��+>͐�=��=]��=[ ����;H�{=z�?>$�"�,��:�$��=q=�1>���=����Mh�=(�=M�ܽ�ݨ<FAk�f����k��<	���=w����*�ħ<{+ɽDt�=�q=�y<�b�qX��G»��=�F��|o��C�=���K��7��=o���r=���z�=]!M=���=͒+<m�<�>f�=��Q=\1=,�s���=����\s��V(��h����&V-=[m�=�?���_�;�E�=��=��=2Ӓ��#�)��=�Ht<R�1�D�<��;x0	;�3��fC���vZ<�����/<7��:���HM�<To���Jb���=�5����;��ܽ 2�'6@����=k������=��c���:ޠ�<�D�=���<a?���z�=xc���<x<](=r������<ͧ >0��=�͛=�K�6j���I�cN��¼=L`�<�<��='3=�o���W�;�0�<~6�=W����=�́�
 =�����W=��=��z�<�=�'�;gJ����=�N�<~���<��S=>b�����!��oȤ�3M���I=$��<�'�p�p=���<V�:��L�Μh=C��P��/��;�<XZ�k%,=Ρ�=��e<��=�e����4�3�=�uL�j����y�;�d;=�/V�$Hx��@=�H\=p�<��2������� >6�ʓ,�6Y��ge�<��q��^��1�<����j�<r���=V�<����뵼����%=��M:ܗռZ���m��Hr�<Ž�<o��<f3�m��<�����P7��\N:��ȻI�<�\�p�=�7��V	=��=�ռÀ��[�q=��?=��<�EƼ���=����x~<�>)�	;-k=s��2(��!��2�Y#;J�<"�=�=��v:���s搽݃�:+Y�<��q��K=@)�;2i=&j=t��;Q���@0��.=G����I�ܹ&��f?�a�+=��z=�����%�ܼۡ��S=!��q	��^�%*���T��D�<A�'�u=��:ؽ�<��=�J<<�ҽy��xo;�i��<�J3=����rf���%=l��=O(�<\���h����E�����Q;м�#
�� =���<V��=��E=�N�=�ŷ=@$=�_μT��n��<�0��٩t=+���Fb<�:=0u�;�O>��P���>L�;���<�:��o��h=��B=g�H��C�"�~�uW���ט���)=��<�}��{�:�i�:��(�<Oi���`�=Nu <t~�<mx=��=$\�;�<q��/o�h��<��<D�V;�(��41�;�r��jڼ�"P��|>�&�8��#G=z����=�(�x�=S����><R��<�5<�R\��N>��3=ۋ�<�TF<�B�����<pk==-Bj=��?�����s����=�Z�<�F=�5S<<
�;�*��/r����<�`O=!>��ѽ���RD�=���2=�P����<'(F=����l=�廇�3�vO��T=��;��`��Ž�<W�Z��������=�O��E����;���=e�F�o��<�2��g<�m�Á����-��K�=�a���M���<�Sa<�D�=�'�����>�t�:�E��ͼg�'��Mu��?;�|I��!7�W�	=��=������;�V��A��=��>�=c��=8B�=��2=�/P=��,<��=d�����*ې<F�<A{S=g���!=�� :s7��	=A�����=�on<��Y^�<�;x=S�0<�O$�����j����<)��=�o/���/=|v��P9����i5�=Th==�=���d�;t�X=��:���Eh�������]����<�؁�S��=��!=Zi�=}�0�����F���=���=hN�8`�"���=�fԼ6pؼ�	�<AT)�j��VQZ=�q���<��=N���}0���'>d٪<o��g��<Jn���ߵ<��/�0���'�=+��mM��-D��>7=:�0�|���J���p=�`B=�/�1�)� N�$�?�A�*�G�y���=��Ĝ���oL<2=�j����p=���|a���<`���ڼD}'�%���s�������w��"���{<��>?Q��n�<�0�=�9)=5>�"�='�<�W=퓧=�Q��zL���)=���=Sߩ� =N�=��>x�;���^�p����r�=	�ĽF#����/����=���=�n=�ل<���<�ټ�=�ӏ�AK#���X=E��<{ټ� �O�t�>�m=c�=���;��g�����ze.��S�}�;Q� ��k?��󉺘;�p.<�E�=G/;]y�=YL�=�З=�R�����=��@���=1�=]�=I��=r�=h�7�FY�<��A�_��<����{��=q��9qvc���	=�B�G�>�A�[��;$=���=�2=���������<#�"����ē�:�W�=H�<�BU����<��M=D
}�g���2���S����;���=.��=Η���מ=u�m��=i�!<�CV��^��]
;�]��$䍼��ऽ�н�ۯ<�c�:T�J=�b̻��=�F缆i��m�<τ�=s� ��S�<C\�<��=��J=��	=1�<��=,X;R�=�=*_v=�	�G���)=�M����=	��=���<t���Y�=.��=�����౽��ٻAt�<(B=ؕN�,�t�}=��<����h�<�)��I�r�4Z�;Vܻ��.=�(޼F�m�S7�*�%�����t���7=� =����|=�2���)P�,��qe��n��T!.�-2�~˫��܀;eҽ�ʢ����;\I�=ȣ�=�}�<�7;��!�<�o�=�l��{[��-�ﻑ���D�<�^)=,��7��]�>�=��_=���;g��=&ą<,�D���=,���4�ȹ��nϼ�����w�=�ͽ/��3�=��%�.�%<X�������O�Q�	�0Z���<�[k=�g=@I=�Ͻ��ѼҢ��Η<�l<�Y�=o��<�����9=N��<L����Z���k="�=��߼ ��o7�=�Yv��c>=�`����<ŝ�=U���<ڽ�#x�9G;����Q[��� =Ui���O2=�P!=��-�bg=XJs=݀=>���J��=
���1�J�	=I��=  �<�٣=b(��T�;��𼼃������)�P e����s�w=��=�Fֽ�Q>�6����=t��O���1�f�2�r.</�=x{�=ϳ��>�g=BI�����e���S�ҵ{�R/=���<G:�;�;�<�	=�)=�������(<2�ֽD(� &�<�+�y��<��d<b e=�-�=%�/=j�=Z�ҽ�#�;���=y�&>I���=�<t��=�=t�ټj�<�����A=;W�=�;�'�f�m����=m�=��=��<�9��.G��J�;A�hh�<��Q=բ׼|�a=UB=W���lx�;�4>V��<���=�f����M�s���=��ܼ��F=�Y{�S�=�	��
��=w���3[��i�*�jLd:��*=c����%����A�����Z/�=r��x7h;c㑼��=���=�0��*�=M���쨽P}�ePB<�铼"=�B�<QR���A�<�i;=��=]��<Rp�;@K����Y=��Pz���c��	��<
\�IpN=K�=>֎�={��%�l=-`=�G <�
��V3��+��26��{t_�,���P�=�f�<�G�=���<��=|B���	��b��=�r=�4��D��:<����<<}�<��8=p,�,��<o8�<@�]�oћ;���;M�<"@K�9F2=��n�ܾ��������=�=�C�=�s�=��<z�m=pH�=�Wh=�4=Nl�mvz=#�=`-�=rZ<p�= �>=�6.��{9<�=��`�.����t��=�N�.U�w�[����+N�=&R�<c���F�����li��޲�==-&=g�>��_=Z��&�=\����=�|�F<�&:<��^���"=T��<�!���=���$v޻.�-���<�FݼD�˹�?��3�o���;�ܥ���=vH��oM
=��}<���=�.�;�̉=0@�<?���/��<E�=��W=I��<_~�:��\�]��.#�Il)<�U5�&>��a���+;�g.<͕(����!�=��ٽ���r�a=��=��#��g�=H�O�=g��U���d<A�k<'m��d��
=3"<C��=�?==݉��)̼���J��E��=,�6��G������3��LC<r��>�;*|:�%@J�1�3=��=d��=�����<%3�<I��=�ć�;��=��@T�<�R��=��4<b�L=}==Imq�U4�=�V������c+=��<�ؔ<*z5<������<�߫�u!�=�u��T=�2�=�=������/< �>Vb	<��ȼ$�<�\$��%=�y;<����>�����=W̗���3=&U�;魶=��=	]߽��=[}뼘y�<��9=�=�8ֽ�����I�:6Cq=~�N��i�< >'�<��:��c�=���;w����	;CFI�f���- {=�B�=zD[<l%�=!n�|E=�>��=4D<����g)�J@�=�~�=<���͉=p��<6Q��aνU�< ��((�=�F=�)����A&�<��=c�=��<���< u{�uɼ<�ɷ��ú=p���t�==�=�kS��/<4K�<u�.=�ى�Oh?�|;h=���x�<�9��=no����(�B��W<ś/=�[<.�<9���M#T�f�A�C8L��Q<l�a=�`=<]<���W�;��
��Ձ��-�:����y2=x��<)���÷�����LX��叼f'��͙��S>&����=�=�=a�=��=��=_=��;8�M=�h�>_4=������=W)�=ސp���=���=uۧ=R��:�ҋ����&��=�J_<t��ɫ=R�=�5�od�=�<L=�f�<�=n� =��b�y=��޼/h�=/y|=�+h�q����"={�J��T�=>**� \8=��<=��-<dQ	��D0<��/<�Խ�P�=_��w-���x��D�%=�Z�=��<�O2����<��P=@���H!=�_=���ʊټ✑=���=���O�<z����w=2�1�I ���<��=�?�=:ʣ=*����-�<�*i=#�����j��<��̽3�K���s=S_}�L0��B�Z=8]�<�*�=�ͼT�r<V�=/zH;�(�=U`��i鉽�����4=�ZF��t!�kw�����<��m<Q/��$�+�蒈�r�p�_�=Oe�e���J���gy��ѧ���=����c�=�Tv=H7=��n��ۻ�T�@	#�đ?=Y]n�y�=��ݻl>�<�+�=�9s��?�=����i�=J�=�#c=���]r=���=yf�g����.�=q�q=n��<��ؽ�ɺ�n�<����5�^<�����Tq�) ����=���=g�S=�=�I��t��-�=x!�=��=��D���<u'<�<��6�!gʽ�=v>�1>=��=@�ԼU�=��p<,Jy���ռ����}�<9��=�I=������>?ɐ=�(��ٷ����jB���a�闗=�C�=�z=؎=�`�="�=�`�=�)��j=.X�=�z=u�>�9�W��3��=��Ҽ8���Y=�=qW��U��!�;�7j���<�͕=��=�R��<������=�w�=)Wb=�=	�=�<���;��=��<_.�=P�
��好���,�����2�E%�=�wO��V�����Z�0=ջ����l����.�s7<Pp�=��e=���$`�<b������<����Y�<�.���=4�X�/�<;b���=c%=��<J�=�-���A˽�{=��==��s<��C���9>��`=J-�<\NR��AżABY=<Ɏ�="�=;3O���;�ߖ��������=�1�<�^��2�=x~�=/�	��eC= �ؽ36���2���>�i�:��3�(@�=��t�&X�=��0<���=�h�JQ$<h�*<�"�<vT��<p���vX=0q�<m));ņ��>�
��C�=�������<�Xs�{�=L�>�'ɼ�i=�<���vW=�v�=d;�������=#�B�S�����>��;���;�v�ֽ։�!Yt>�w�K(�=��<�
<���ed>�>�UT��<T�N>v �"���v�Y��;�=�:�=iR�����=_����P=A���a=ƻ�x�>�/u�=p�d<}�p=��t=JD
=$R�=�LH��0R>r/*��3; }5�<..>LC>��="r���G�=��>���=ޓ�=*��=U�k<j�$:��C=�K>2�n<8�I�[C;	���y���=��5����<����3���,=ֆ�=)ג������<(+�<���<�#[<R��=���<�3���\ ����V�a����=�������mB�P��`�9`�<\>�)�=�\=I�<nh�$8�:D`2�?J<wmw�i���2�c�����:�U��=O<����7~�I_�8H�x������$�<��$=PI���<=$U �O��= re;"��j�:S�5�
��=���ǣ�K��G�; g+:g���2�=6���k>�n������={�9��/��̽EL�&R@= K >�9o�� =H3�<���'4���=H��<6���8;0�=~�/��0�=r���9����к?�<ij=�n=6��=��J�W$�df�=W�;�eѽ����#F-<���Dդ=W`��5ԻKt�=&"�p��<�,g=v;�����2^߽�Qɽ��/>D���@߽	bv; r�<���#���h�}�<��>0ɷ�� ;`���Z�����<�ܽ!�x����=Ȟ+��S$��6>Y�=�Y�����:޼Ͻ6��� ݑ= ��&�<5��~_=�n�<[a�=���=F�'��IW���I=4%˽7 �;�<[�=Ќ��~ 3�$�>��#�(;[ ;I�n��K;Q�=s�:=Y;��?S�<�X��rf��H�=Z���'<�91���lϯ�LՎ<�zU=.}����Z�[�H�f=�5�i^�=$���5q��罽��U�+���Y^z���=���=[�Ͻ$�O=�,�Dt=n�ϻ�Y㼱�y��2�<�uM��P��Ԏ�sN���PO�д�=�=�p=t:��
��B�=(=�7û*ی��|����$�F��
1M�ඎ�5��=���<�
1���<��ڻT= =��>I�1��<������=����ge�<A}����<���;�鏽��<U#=�<7Ƚfh��򓍽E"</���[�<_�i��[�<4�_;���	k=�%B��5��S��p;�>��=���	C�<r�k=�=�s8��E���c�=+�ɽŴ�ULB=�c�<pe=�L�=�9���c�Pgн�wp�Kۆ=_��7��&+)=��m:��=&�P��Z-<ɠ�<�Y���:�<�4���je�[���_�;U�μ&g��6��Y���;ߒ���*b�;����=ֽF�p�=�S���=9��=w?ռᕈ�2��=0l����-<��2H=�z��+-����!�L;=��Y<���Q���<%��<G�4=��,= �-������=Q���<��b=�>@�5�;=���=d<����1��H�=\�<H���ʭ�=�e~=C7w=ὛN��6=��q=f�g��G;��v�''<���=�=��3=�4��ǐ[=xk��x�L�g�N�=�q��_4=��q=�-��m�j~<Y� �ás=?�x��<��	�J��<w��=;�y�e�����0�I#�������<�f?��%�=.�=��+,=�f'�[H�r����KݽrX
>���=y.�=���;�W�=���=� �<,�v=�i=dF���'�O=[Iu��N�;S,�<ԓ=�'�~�=�F<ӡ`=�Ä=Ȼ>=��D�`�q=d}�=�Dj=Ta����=��;zr>��=5v5�E���1��i�=�%��삽B��.�=�����/J�;�L=7�=���;���<�Mn�xͽU��<��3�t���<�=`>��=*�=�粽nǽ_ԓ���c=no�=A�����H�e--=�焽�'�Y=S.�;ű=+�C=o���������=3��k��<j�>�s�7=u��߈��z�D����<��b=�DZ=��	���q=�IV=b 輭zu<Q׮���i�w��</���W� ��lü�s�����&e��BȽ�3�=�E�=��O����	�=�E��!Z�yK�=y�=��=��n��"?�=��3�	��<�޶�裙��S|�D�(>�h�=�!9�g꼬��<��B=5ؼ���=Ï�;b�p��F=�-1=�8	=�b̻��<=Y�F�BxP<;:"=��5=?s=^��Ac�@������=Aα���	��mM�\�ּ�g�=.,�<�ɕ���T=z���a�<��7=$Kn=@�C�����jq��<�s�_O���E��'Ͻ�Gɽ�i���=���.b=O��=J2I:�N�<S�P�qƽ�����r9�m��=ϵv�̓���<��
<6��gc/���������,2�=���< F�=��+����<�M�$�l��K=>�a=u1��q��<��4��x=���a��G.��F���3��[���D���>�|^����¬T� �R��i�<�_<�+�=���1��<�Y�^2��H�:�>�<�Ю<����,�=j,>Ň�=z��`��E�=��0�Т�]Q����*�>�2<g��SC�=���R�	�t�W> �<�3���>�I�4@�9"m�<�%k=�ᨽtr4�)B齑<>Bb�=�t���B>|;'�{�=c	�d���^:=!�����;$S<>��=�Q��c�
��?R�xv<qy�;X[p=_6�<Q�������=��=�P3�>U���=cTs<�<���;,��=v.v=:�=��B<��M=EӼ��j��T=�qJ�R���U�>����<�V���!=�c����=�`�=~
�8ح���»، >{���a:��h���X���yA�~�+=��;5���i�;T�.C=N�a=�s�����C<�iԼ��=
��x�E���7<`=I�4�1&k��u[�����IZ<Y=Ψ�:����V�=�e��P=s�=�3�'=���;� =�b��_�<%1�/���->��!=f/��`=��%=A�� H��lD=��c=歟��&;*)=��)��=M6�<�$�=��C`��7ņ=ω�ꩻ<@����	���>��-=����PVu��s���oE��2�=^�:���<H��B�	����4��i���a�5�[6h<�>����򕂽�< �����=��=��<�5�=�qͼ乮��ߖ�r�{==�P���7��� �7����=�+x;A�(>���WϜ��0"=��S�ea�=��Ｔ��\l|���=h�����s�5<Xn�dRϽ�ǚ=��=`�Ӽ�`�=;~߼⽻=?��ˬ�<1ns<bSg���;�J����Žqx�=�"P���ϻ���S5�;6|"<��W=\��MƱ=}j��Qڽ�nk�|s=�>��⑽���8�z���>Ʌ�J�#�˳ȼ(�=��λ6hF�Wj��j����Y<���=�"L�,�	��d���Hż7]=s�μ}�/=��0��eC�p���t>�1=O��<X�|�oټ�V8�><a=2 �<x<�<��C=o:��e�;�Z�
��<�B>�4�<�����
��≽�'⼳\d���'Y�<�,¼�3�:Iv���V=��i!P;�3�<-.w=��;�a�?�!@=f��ؘ��7�\: ��<�M���F�;��w=+災9�#��<��~�93�3�n��q�<�'x=�?�I����?^����;�/r�)f<=��z�<�k2=�k��M�\�J�v���9����:�<(d=�".<�Jּ��2<����t$=%����Gz=�=zBU=MF<��<�7<�;=�3@�|�K;	��;;��=N�Y��ϼC�j>��B��x�Ѽ~���N<(ć<7*�;=ى=���<	l�j6;=[�h=�� =V�=��=_�<6><+���h6��!d��[���&���*�b=X�E��b3;���b\�S�1<Q�G�9S"�����</T�<��@=&�ܼp�ӽ��</!��Wf�<L�'=��*>]k��,�pA <��m��7>�@>�^�ֈ��l>5������=3��<\�}�@L&�\�=��3>��+�����<)f�=�N�=����(7�=kB�t�ټ�:{�j׻��!>w�&=Ԑ0=��*���<OJ,>�(?>a�"��5?��?�=��~����<��<� �E�`=hE/���S�>������Ѝ½=�=��>:0
�R&<��s<4�e>�=�e���y��K�<��<3��<x_"�j��=u�_���4=�u�<̵���ѼIM�u��<1I����7<5�;�U�<����zr��XؼoE	�u�$=��=�j<�����v9�i���B����F<���I6�=i.�u��<HI�Y����&���։��w�;��k���B��	�<p�T���Ř��*$9�~|����򽽗��P���&�0��I�[�H�'!T<}��F=��=��;"��zu���7:��0���<�F)=�Ϋ=�%�=�~ɼ����{������Ym�=�=���<�؁��򓽟�r=�#���ђ<�e<$==j���E��e��<5�<e"!�ݶ��^�==��B���<P��=�q��a�<)ॽѢ�<̢�C|<m�>�g�=���I�����U='V�Z�μkDe=�G��ý�@>�G���@�=��;
�yr���,�(���%��84@����=�.'�%��/��<�{;�۳<:h�<$j��!'<"1Ƚ<c�*���=~���tS=�s<w�W�=�
=�)ἛaK���,<%q�=�7<��h<�!=���=ȵ=vq(=�=��=��G����=�;��6��w�=�YT�}Q�<;�<�����S>��T=LQ�<[�=ýY�=6{�;(�=����r�>��G�n����~3=�ix���=��,<�~O��n�:���<� ���w�<3�N���8��!�=� =�̃=�7��}@=�XX<�x��!�
=��7��](=���=��x��@e;)�#�.V��2���;�<����'k=�=��<�.����)���c��Ǎ��r�=�*3=K�� �;}�H>����<=�<��<��=_V"�	� =s6�=�W:�V��	���F��˭�=<  �����,���㽂f���{
���>O���nݼ\n�<eִ<��)���3>��]Y�|䪼E�㼶񪽐B�<�¼Ӗ��(�O���`*K>��
����s+����m�=:=No;������<r�;��(νy�1<�~���A��CV�=r�=��=�=���<�_�PtQ<fα=������佼zܛ=2�}���<�s�]�]�T��SY��l�<�D=�F��=n�I=��;���b��<NQ?���l������<-�<'ڼ��T;&�=WI=�t=��<���<���蓻;�P==��;��0<��<uc����'=��<�����b�l����
=�<�&==Z ����.�-=�������<4!�H��<��ʽ6<"���>��<\v���2��X��·):�;�z�G=2\ <-���(�w�rj���)�O��=W�T=Bנ�?���������E��;�=d뗼0��{��=I���[<ÄS�I�$=F*�=�	A�ٸԽ�c��;���4��=K�
�{=�TѼ�N=C��Z��.Aż�*�=E(�=&%���|^=���ڴZ�hA;�7=��=
��<
/��Q�=ȴ�=�W=CV;=�O��&ܻfa(�/N=�H�;瀮���6<ܻ�<� �=ΰ?<9�=��<�3�=J�Q;| ֺP�=O
�<�=ܨI=��=r�=��"�_��;{s"=�{>C����\=&=�=e:4=7H!�b�K���#=��
=���<kF=ۇs���=�����<Yp<�=��,�?p=k;Q<�S����<�`�=�}"���<�ˀ��~U���B=��S;\c=C�7�z�=�>>���φ�,����Zؽ�;�=p�]=��@=�q�}⁼]��<�s�=-���ه=��p=kk��ed�<��<�;�<�-'<��0=�Q�U��=�
t=��<S�E2����$�A��<.�*;��q=.�=��j��u���
<M��=�0=V����;���4\�<�#=tb�<�n�=��=��p����9H�<�Ŝ=,�]�D���d���8�=�KP�lR+=z����u=j���N��<b�P=]���<��O<*=�ø��=� {��m���Ѽ2��=}]r9h�= =��I4J=͖X��� �=��;�;>"<Z;�Dм~*:=*9�=goW=�f	>�g�<Ð��$�@=Ba<=L����Iv����=����7uV=�y>�{�<59�=�3���*��{i=��<�u�7�0��!�<l>�[�=z�=�)�=6p�)q@=ף�@��=`�-���P��>����<iK�἖<�qw���=��޺g�,<*~�M��=Ӟw��F���=��=�IW=���	`�=�\;oyb=��=�h=8�a;b�x<u�\<�ڣ��57<�E=TѼa�;񋕼�U�;&�)=~(�<�=�o����=�ݽ�x��t�;��h�}���]=_�=Ī5�&w������߽w)�=�+�=��@���>�t��=���=h<�C/�(ں<$pH=�H���"��LX��-����=%H���r���=���=��B��0^<{3�h �!�߼hC��xǻŚ�<DL��K�׽C �=��={�;��f��"���=��<JH�<޿<�c}����<ӑm=ǻ�|T`�h�= �ټb�(��ֱ<䡬�����4�"劽�=�� =��<z�|<�A�<1Ž��W�.f�<H�=ar����<�i���C�=NH�=]�=(/K�yhM��=���<Ov��Y���Z-�y:��鳛��z�:c{<bʡ:�����D�ԫ<}
�=d���$��\�=�H6�q��:T��=Q.A��I,<<&�<¼�9�^�\n=�_=>y��;��ڽ��=�x�=5�:��=3	��`�=�A1�!��<q��V4�<q]�<Y5=��Y=�׷=���[��=[���J#=��X�!32�	�D����h��<q^'������Ҡ�<���W�L�S{)��m��].<5�a�]�;�m�;�d�c3ռ��"<O�f<�j==�K���v;Ӕm<����4g=_5���@z���R;>V^���;�h�;����2�=ը<6H��f�=��,�rf��@ѻ9���{˙=����=<)���.�=��6=Ͱ�;��=�?<�`�������g-<��6����:�[�sx=:y�XPֻw��<�?�=��=�Y�����<��=���:-���k���$��+����W�=Pڽ������<&�;����˄<�E�=?��<nB���9p=!��;��ɼ�AH�$�P��+�<އ��A��<Z�����!=��5����<:���j߿=��@�p�[�����g�<-�n=��<v�U<�n麐�P=8@)<�j�v�]=kE�;eBٽt��H����eP=�勼�7�:��=��:=�ǼLd�����$@���<!,��=:e�=ݮ�;�m>V~D���8��=���=��<4`���(�ViO���<��=��n��*>z�%����ۼ=h�=��U<��9���=�&	=W�=|�r=�ֲ;u��3�_=�#�O�?�����`�,=}�	�'f��:=ry�av�����Q�#�h
>jo�<o�U���Y<ۦk�P�=h�<q�<P�X��u�=X+��s�;l�`=5�<����d��=���?�T�Uov�Ơ�<9��:��@=�4h�����k:��_��{��醦�6@S�I�5�{F�T���}*�=���=��<!ac<����|� �{h�<~����R��<���<`�=�yg��<���"
<-�G��� �{��;O�L=�=�a;ػ�<j���e��;P?+=����"<^<BHv�W��Vw.�8��<�9)����-�V���
=�F��qn�=6줽��5�XᎼUp<��ļH�=H�w�?k5=�;���Q�z=�g{<�h|��
=�v�FF�8�Y=w���`�e=qY�����=����Dɼ���;�$G=�T��"�&����;k�c=�w�=�����ẰJu=�<	�vD<<c*~=�����l���U��"#�<t= �_=�R=�2�<�� =�M�=G�g<r=�脽fm����k�<�F=�'�eD����=1��Y�;�B��]< 
��ܒ�<p���\̽z��<�@� .ۼ��O�M�軏�q=!*�;O=UJ��G��?b��*v��42�=��мgL=wrսװ�=Q>��۔9=Q�<K��=tQ����Ҽ/{�tk=������<�P��w�z=˼���=(<�=��c���=�g��<����.�<�F�7�l����<�RX<�;ͽ�_<�:a��<�����ρ����=�	i�S�=M���Շy=�g�4��=,��;!l=�a�,�F�&=�	��^ؼ�%�=�1#=s�H=�K<�>׽�^�=���=&o7�ևH==���<��b��e��81=���O�<n�н�����Ҭ=w�7=�A�M�=�z�=7�=b^�=/N���4��� <>����1Xǽ��R�z�9=��>.+�����Tf=Ⴂ������V">+>d>��+=R5��ʘ=;�+<!~=���v
>A�h��nc���B,=�ǽ���2�f�s�<�s��)3�=]/<�{�=t����HJ<�<�=��q1=7}S=�6A��/ϽE��=��>�#���Q��V�=���=�r�=�^�=�l@�I6=59��O�=]2��h���+=���;Sn��4�B�����������`��<�u�=ͣ�=�J�<4j\�
{�:X>��M��`7��L+V�k=��R���@�o
�<�_T=*����L��=Ro���a�9�@�=�a��4DŻ=�<�d�<$!=i8�*�=���/�=A4�<�ʸ=w�ƽ�Vv;
F��>}<\�=A
�̜7>�6=�i�<��=�~���P�Q��=��D����=h$���А�4���٤B��7�=����gϪ;��+������qb�o�$�Dȝ�]S?=M �=h���>E	�s��{t��B�yϤ<8}i�e�b=�F���t�= 6d=Mr<i����=�s<�CA�mx��M� ����<ź'�r7=w)7�u�R���=^�������ѱ�$�<:)E���4=�я�[|C�|I�B����=h=|]�;�Oi����<���?V�<`�U<3X��Bܼ�+1�����[=?�|��Q���;��Z<$FV�����/����X<~��<�X�=��y<qb��������<����"�<�Y�<;��<D�c�ёR�O؞=���=�)�=�Mܽ�\T��|��A~�����;)��<s�<ZF��ߐ����}�|}Z<J�<��޽��Z��u�=���=ۭ۽�ى�e����[����"�ϊ�Cv���<#�S�ڽ=��<�#>��Y�O��T;�ƱZ=�z��٘�<���<X�(�Y%=%_����=��M=��t=�����'=k
[=:A��j�И<C�*��1(<3p�=N��	Em��<���]���=�2`=�Z<$�ּ�g�l�-��=�l=�c<�q�X�b���
��=$�a�%-�<��<Aļ�s��0RS=���oB�=��н��Լ���;����EL�`j�v�<�c�=#�Ž�A��|M�ɮ.=	�n��:��&=�v�=cXѼ A�=���v�,�;�R�<-����T���\������!4<d��=�����S��!�fվ<a�=)4��>�*)�<Ǘ��i�����O�WC��G<6�f��z�<TF���g��;������;h8 >��">������;E���=����Eb>� ��KW��-��T��9�*=L��>�y�=�u����	>	/>���ud��sн������	�.��
h�{ͼ}�6�~b���R>q�ȼ��;��&>Y�н�h=��>��J�\[�=�s8=a�>D���7����=͸�=�If>@��</�<��ڽ���>gދ����HI]����� =��I�6�>��4�s�۽;�;�s�<�_�<�%��O��=�&���.4�ȑ��l>>����G���w��tmS�	��= �=�A=����|�j=�a�</��<��=�D<�۽�GU�,�P=؏= K��a����<���4(�X�m�6�ҽOU��A��� �O#�=] ���=�躻��>=0c���#�=t蟻�I}��!�=�}=�j�~������<ʦ%>L��N�X��F�=<�=��g=N�Y=%�<m�==����/I�m9��D>����<��=�����<��A=:c�<F�=$��<����c^<Y�<� �p���پ�k�=Wu�=3l]=��w����=�Ǉ��|:��?�=t�d<E���Ƃ��tG=$|=(���j���K�<�2c<�dF�x�<w� �d"/����d<U����=g^ =���z��Տ�<|7��x��+����~�<��=&@�ɭ�=� �oA�=k��=aI�<*���'=�� =fM�=~��=V0�pU1=���ɾ�=%,��=����x�<CAF��;�Q̴=?[�;Ch=��	�EKټؕ��pV��!�=��6|�p"=X޿����=�v�;��@=n}=b��� �=��ռ��1������1ѻ�c=��q��z�1U��Q�=����>=�ߖ<��G�<k�<[�y���=�漩��=`����?�;���<���;��+=���=��;��=C�[���X�c%=B�,=N*=�(�<��=lN���i���p�=�	<�S��=kʡ=����2�<��'��3�:%}�<���Y�V������2���=����^�<�3�=��н\#���;7>�x=��'�x���9$�e���$��=Q͂��n��J����=�����<�uU�tk��_�������`.���Ƽ�/��ϴ!����ֽ+s&��?9=��)��F�e6�AI<f�< s:<I=�*y��ꅽ��<��TN=Zp�=�x=���.Qڽ:�*>�s<��<������=\����ͼl>���<,�(�@J;���A=aw�=8^����� 1��;Xݼ�w�=ZT�=���<6��=x�H���=/��;V\�=�B=������=rƫ��q��5IW=�L�=����y�v{=���=#f�������W�=�Լ�:�Z>Խc�<h���ջ����ґ�c�<Z�=���?��<�=�Z㼁�d=J[�*���6�=�\������l�<��=�F�=a��f2=i�;��~�S�����O�`M��?�<^1=� �AD�� ���9>�Nռe<�D)���I�=�=ؼ���<R|7=@1��V��T��<�D��'�=��u=�{�=�E�A=�Yƽ�%����'���=��r;{e��
��=u\3�c�<��Q��;��=S�Խf+���1I<f3 �����#�R�t:�#=ù�=Nu�����������[a��
�=P�=�>~=No�<�dнO(���H��=/s�=��:�6���B=�RC��Ÿ9/a޽����lu�=�"�=�-�r�'=֮�<��=���=�Ј��C�;��<����'�rO�=��=�U聽bӶ�Ƶ��lR�~}�
}�;���Bؽ��=B�ֽ�u��B괼K�e=h�;_t��6=�����\Y�����w=xa�=L���ս���<WN�VYS=wx���V�ĸ�=���=�.\��=߷<�dἼ��;���;���=PW�q&<���(��=���=) ��;%�w�Ľ�RT=�i =��һT�����D�n�y=7���4=p�]<*I��8�M�T�|�SX���D�7e6��q��!J=9�<Y�#>Cp|�_����h���E�6���/�=%���ݱ���̺g�ڽ婩����=\��=���!����a	>�������Q���=�7=�$z��n�)�B�iG����ὖ��=�D��^�>2�>/W�����=7�=W�ὢ�N�$4�1r�=�kݽ��Ҽ>��D(=t�>	&ǼP�=��F=Y�'>ٿ
���I=����8Sż��=�(��`�<4J�<���� >zo"=��=(H�=�eI<�:����}��hl�=���<���ZP�<꼝�����g��=��=��_<r�������d{��(�=���=��;N�����=�?$�~�m<Z7�<��=�J��G�ݼ$=�;i��T���О=��D��d�=�VK;+�=U���Ƽ>�-��=A�<-���w�>��Z=�����k����������;�����*�=A1��.<�>=`�ƽ6�<�f�΅)>�F��{�
�{=���ռ��=oWT���?= Nh���1�&k<����ǂ�;�&M<�敼�+�<md��=T�=f�=��N<͟>�i=��B=�/0�L��<d4��0�����z�7=ۇ	=Tu�<��½�zƼ�D��XB<�Ԍ��2�9��=nd�ݻ�B[z=q�x=q��r�<����&��<o'Ž?Z=@<�uB=Ȝ�;{��	ԼH��������M�d�im>=9�a�ջd=(��<v���P����彉@Y����=�涻-�R��I�1�뻳Ə=����P�n=��=��=G��<2-<1c���򼗝�=�ژ=�9�=-=�]�=�lB����<��=X�=�]	���?���]��`�=��:G�&=�x���=޵%��Mg���Y�A����=< ��G=nn�=t�>�t�j�u��5��,nS�[0���0���c=�Z��!5�Ǘ�:2=�$�=�))=,�#���<o��=�b���<����y}�=�n*�WP_=6Ƚ}������<=N�<3N�<��7��T�;MH�:X(ļ�=R��<�q��f~����=��D;�������<u���iN=�����Ώ����;��<#�=F:�;U=���[#�E2=�}n<�u��.ɽ#|%=V�<WU�==�
�<5�G=�M�<����Y=,8��K
0=d�r��$Q��Q�\�B����)��<�ǻ���ż�8�=��=	y0���;��?<�o;R-�=4�����<Aޒ����=�'��c<��<���DL�(�<�J���=n��%�$��j<yU�<s�%���Z=��=p_޽	��N�	�9˜=�b�;����T#=��<
�!��嵽��<?!n=?>b=�w�d�=M�R��}��g�=��<�=$��<]���?�I�,B �H��=�@��Ď{=��=�μ��<A(=9��+8;���=G�e��49�~;�QCW�����"G<P54��P��iI���0�XT�= =�&!�����x;�Vz�;Y�= l/�8���$=ɰ(=����@��N"�%<vJ��0��X�;D4<(㸼DW=�5��UƸ�q���ah��1=�d����)��W�������6�o+�=iZ=�o�1��=Ȼ��9���<p��=�6�=A�:�l:�����={�^�K����gJ����=�ݽ<&�	����y�=k��< J��I�K=��ּN=5=�v��d�>2X�-e=z�Gk�<Ԕ�=T1���T׼p6�����=��ί�<g�G=��N=�tU�&a�=��Y<�G޻�E��C=Ӿ�<0���<�᲼I
�T�\=�z�=T�=���=;Q ���a=�>&�V�ݻ:hY<;ɣ�š<�i<0���ɬ���d;�'�y�A=B�T=�-3�Ԙƽ��l��p0����}�������Y&�Q[�<M�Z<a�<+	=G��R%���!�<�f�=}8�<Ӽ�D���"�cṻQ�<� 7�9#P=�-2<��=��;�G<Ԕ]=��%�<�q����<�HE:�H�<�ߪ�ⴑ=�Z�b-��ZN���`=�6�Q�<<��=}0=���=˼�=�a�=�K=H���>g�=��	��_?X<����`���B��4�;r�D=����J=1�iz=��*�<��;��,�ڐ�=�L:�cY-�z��<ʌ�=\j���K=������ڽ�3��Q�-�v���H+����4�ͽ�LG�s��� �=������i��V=a)=l��o�i:8=���<Bu����˽ �b�r*�MQO=QI�<�<1�;[_=?ڍ<��'<�)���[�8����<�ؼn��2�=Ŋ�9� ռ��=l[x�(��=zx�#1=��4��!˼�I�;xG�<hu��^���m��=]���Y���;��m=סk�Щ{<�x�<�һ�v��VE�=Q�d�Z���ϒ <\1�� d�=�g�M�����=#<�����������;K�<�ػn0��'�<a����,=���<l��<��<qP�� ^��&�=�:�;��=4^U;;�>#��<c�=�6=�肽�xɽ��=��;.�2=L�<RW =%6=�*";TS�=�|R<@�K=��7̎��</=-��=f0�=������`2=�H�<,�=�4Ľ� [<_�t��.��x��,"���M�R+i�j��=��<�����$�<���>�4�E��<?�C;��;���=�|����<ܼ���<��ּ���=Wȧ<�;�<�B�=��[��<�G!\=���;>�Q�S�G�T�>��.=�h=��_=L��=.[=u��<���=m,��p�X=�=�-=��;t�=�;���<	A�=c��<������I=rH�����\=}�{<U\���b��n�=n��<�Z�@)�<�&>�P=�E��V<�Ƚn4ѹ��=�������=�������KN:;�?=�2����gʴ���<9�=��j<��˽�����t=�q<�%:������»��=��=+4��uŁ�Am?=�t<�V��\�#�<I~N<�@=c,<	ż^�=��<�=evh�m&-��=�;��=���<0/��#�&v�<9f�=�AR=
��u >Id5��9�xӡ=�3�;�˞�p4�'	�<Ś����B=[=0�; A+8�t�<�a��M�����Lm�=����K�;�<�/=D����=�h�O��=Xƃ��p�:�؝<_o���ъ��D!=H�=1�=�(<ˊ �绡=L=l$J=a0;`���=�A!��<�k2��u�<��V=r}=OZ���;�=��<!٬=���8�Ƚ�Q�:du�<�I<�޻c0=�%	��� ���>�0�с=��+=_���y��Iy����=1ԋ<��^�@=2jt={M7�'�J�����f�=1c�~�����)<d����8� a����O�H=�c�<�9���p���N����6=�
J=��N�{4�=�Ԑ<�<Ƿ7��=�<�~�<��M����>=!�@=�=�=q�����G��~B��.=+k��0 ��[��<���=���<�u<���n�<��n=����>�~�=��@�f>�;�j��3����v!=�	��j5�ٕ�=;�a7l�U������=��<�r$�=?�=5��=Ζ�=�>=����V�'n����<~î�ê��9��ƽZ��.���qS<�W�Q��,V;>#���������~<��,����;�Tc��Ń<l�<�3n=J�)`�<6��;������v<yZ�8]�<���:d�=���<�ỽgcQ���=�����h=0���^O�< �������$�<��=�֘=8Z/���f<�ʧ<��<µ>��t�ߪ���h$<�~޼�V_=��=��=����y�9{�=.����=B��;G�=$&�<���=> <��M�ʟ���!�<=����b���=Qe�=��ջ,��8����<��h�=�,=�f�."��Ƹ�L��<��<�\�=%��<C7���u<�2p���>�I�;���ht=~6�<��1=��|��#��[JP<�l;�Mx��+a=E�0==�@���Q���=)�(���< ~ >0=E���%>�=W���V=�'����8��;��=Ա4�M=�E��=�0�cC���#�;r>�=�O�<���=�1�=��¼�1�5�<鶞��ד=}����z\����; �ֹ����{<d����V=���]+�[���t�<�QC<et��,��96��;�KI���<P�k�=�>4l��gn=���<Ł����=dr�=�85=��׹���!%8;*Ң����0����n��N����]�<F%�����>;??���Rf=�ٽ�1�%��>� ��iB�98��Ĭ����t�>�F=�Z��N-=�;>���
���W㉽���={s�<ӿ\�'���Q���;xV�<=��=���_��=�K�>�.����=ف >lNƽ~}=����%�>P�x� "�<q�=ܢn�MW�>d��=;��=��{�=�>%)���m����B��M��B�a=ƞ}�wB�=�2�<𸏽�� =�5=����AѨ��>��H�Q���t��z0�<ܓ��x`��E�ȼu�w=&/�<�ߒ=��`�6�=Q-n�x���ΉE<Y���U�F3ֽS�=�|�=��I�y��<='=�׹�\�G=>��o��"��x�[�K�U���	=H�
=�:J<�f�#o=1�8�Ym�;��_=D�d�t��=F_̼���=�wD���$,�<��>�c�#=xD<־e=Մ<I�[<R«�%����3u=�^Ž[�=V]4=��=tNG=���=.�A=��>���:O��c�D��r���O�:�q��y�<jY���,�c#}=����<��s=���<w��<U�=?+=���� +�P��<�*=U$&=z�������䯻Х����c=\e�a��<KG�ز>�I���� <r����%x=�Yd�^<@=����`�<���=^�=�+�	2N=ǀ�=yb�� 轉DP=�q��y��=�`R��N<�:C<붘=k��7�{�Q<�`�=��E{�<1��?y׽������U���7=!�=i��=��~��-F���+�s�������x�;3�o!�� ��l�ֻم3�F��=!މ=9����XU>���u<*A��Ყ�
�"� �e�<�#z��飼�b�Z>�-ҽ��<���=M���U={=�f�=�G��掽;���*O�=���
�5={a_�Z�<Rg�=����形�U��Ȭ=T1�(����F��d[<A�=X۰���7>
��������r��Ř��A��dT�� =Pt�����
=7,=xPJ�ҁ�< ��A�����k�=�ӹ�� ��ϻE;���As���<=ir�=�GI�EI�\X�==衼FK=�G���̨<e�;uf�3�*���<2���R<���;� ��#�<n1�=e���D)����z`�<?�W=H�K����=������W�/Υ=�x�Y >��=Z�>�+�<�=�M�=39=�f���}�5�0��<6@��2���n��h�>{�t<5�=4#�<��:=v���� =�h�j佽��i� �\=�8+<<Jn�68�>V�<&6�<�3���<��+Ux=u`{�f	ۺ�a�:.�<�zN=>��h�<�*�<��v;q�=�T(<�Ӽ�X�;ȱѼG�;�h-<���=�>р�=N9�=�=����4<J��;�v����:�
-<�(I���(;Zp������<���_���^=�=ETм#��<1=жF�<"�<�j�=���}��҆{� a����=��=�>b���K=�kk=b������!.a<ʥg=�C�=�u=��<�6����¸&�F�=d�~=e`�/�=G�ڽ1 ����<�Ś=
,5�}������=�{��;��������r�� ^�=�S��ա�Ȓ=Ɉ��<H`,��1?��7ͼFH{=�����tW=/�׽w��=�(=�v��c�=C}<�)\s�����F=�VO=-�=��#������=3��<��=m&�;��<ݬ�<��=��ܽ�婼���<B)=L�g�$�q=��;��=TC켕+���@�o,�VS<;�E=�.2;$<�g=��=���ud��7�=-d���%W=�7�Wk4=+#�<����Q��t�rF�#��;�b�{�˼�5���4���I�@�k��m8=�u���<���:&�_=-A���e��o����0d=�=�%�4X��M��=a3�4�}=����'��<��!>+�뽗��<ற�E5�;Ζ=���<�w;P�t:���=M��?�P="���y�a��)=<=_,��ޠ�<0)G���=�l&��#d8�q;V���wۢ<HR";�`����=o̭;�N�<�e�����<��3�{�>��s=�M�<<��� ��C�;={�ͼ�L=\)'=x���s޼�.%���I��cO�P4���	�ؠ��?�52m=Z}�<�bW��g�=�:X=<�
<%'�<�x\����=is���&�A.���=�=�y-��2�<̴�O��EǦ�����8ؽF�������td=��==ߥz;��<�Tw�j�r��9<<>ݼ�)G=�-�=Ib@�'��<�x�<3�0�
�½�nμpc�<v���'!�=z2=���<p�>���}�����0μm(�<H��͓=cﯽr<Qo��s�<;���kf�)��ɫ�<E�~��ϑ�WB� =D�=��=�ؽ,nż-g��9\�<�$�9�k:M�T=��D<%���=���=W �=}�=�0��=�4<��=�#f=æ4�E~�������6<��'��e���N=U�;�'t�"'�<���l�=W��;�����">=��<��ɽ��<ɝ�=E�<H��=8�<���=O{��7%�<����gG�=b�!<�>%�b�W�2=����v:<!�R�g�󄼽9�=���O:`�p^��w��".��o��WM5=�ǻ=�R =���U�(;�*���>M��R���?�*P�<A����E=.�A����pѢ<���� ͻWV<0�<������i<�q���e-���b=��9���<$�Q=T���b��=�k�<��Y=���<�q��=8>�};��n�Q�C=`t?���$�λt�:=xY�<l�=�E;����>��<�����B��S��<�>B��<=QG`=6�[=P����<�B��]�;�P�"z�,^h�('=}=��.��+==Z��<��ǽ��==U�=��V<1�V���=9��=08J�	ݏ=9��<Q�˼k�:�l8;���<u�>4ڠ=W�w=d�?=�;";5%;����l��;g�=>��=��E�f�y���\=�n<�|=4穽�U���gր=y;�<gZ���a==1
>o9.<^��<�e�=�_=�N9��}ٽ���=C'�=b�=SQнA1o=�cY������8=�<�|=t �=�8�<Q��;L�<�ۊ�V:Z�rD���y��$?<�-�����=@x��s�<�5�=�-�;���=���f����(:�=��=wy�<gkK���=���u��=�1��6�ռ%��j����y�3P=��w2���=-�=0$=��=f�U���(=�c�=q������$>S�e=vÙ�(�:�"a=+,~��e�=�{��">��J����=u���Ϛ=��=�bx=yu�Su=�[�=g�=��F����@=����XQ�<��5�Eyn�i��쎡<1N�{ۼ<\�<�S�M���'�=k��q��:���;o�=�$=m�E=^*�=��ۼ��{�ё�=-Dh����E�^<�K��i=��=��%��ϡ<<u���=�枾�����E��B�=�-=1����=D�U<Hh���0�<�r��ī�=�;I=USI��H�<�d���<r=/GX=rE~=I`����|:	9�i(=�v�=��޼'k��G:���O㻵I�<��;��/=*F���*�=y���>̼�`�<|ș�� ������u���2�p��<e��(�^<��y<}8"�r��=�E���<��<TǕ:h�P<Oh�<�B=+Kl��Z�����=�u=�W����r�����;z=��˼�m�<�*>��:<?j=@��<�j=c�����<Ңm=m��<,JL�L�=) �=^ƾ��?���{�x�=��=˲C=�ÿ��T*=��8�0���\=�^>�f`=.5=��u��Z#��W���`=�>`l�;�V��	�ϼ�k�ѤսCyѻ�lƽD�=l5�<�(=B�^�n<����Ž`�a=�j��#�=�]��?��<�J�����=���=��= C&�w!
��*N=Qv�y�<(���=��<ީm�Aq�=�s�D����<k �<~���ˏ�<��H��S�u�Y�*<�ڽ�WG>n�=?i��漜ຽ�L�����=��;�T�=�.p=~��$�<��>��b=7�c<�	��:�=�hϽ�Bi=�bN�LzR=�[̼8ƺ=Anw�_0�K��T�U�}^3��>��?���K>v4�4�=Ջ���3�:8�i�ݥϽ�8>�]�<Q��=�N����-F�>�'�=����N��9���=�G�5�Ƚ6�m= ˬ���"�K;��5'��Z���=�=m��<���<�ލ=	�=3X��p$�fu=;�>��>��ټt%���Ϡ=V��)o�s6m�M���'��<��WϽL��=_�=U[=S��t�=�--��^Q<���������f����:��3�V5q=�༦��(%d��߼;4*�BiU�{��=�)�<�3���^=���W9[����������<����Ȉ >�`m=�]���Y1�C���i�_<�<�=���<�68�r�f��ּ)���c~ҽ�f���N�<�<%���YL�<B�W=�`���T�\Mw�+W���eo> �Q=��=�[=1�=ᏺ<���8���;�J�<�=�63=(~9��>`�>���kd�������=���xF=���<���7i�=�'�<��=2O�AQ=���3r=��q�M�I:0�s��\b�Q����;������!=���3�۽r�;��� �m�>�|�NE���< �=��8<��r@;��C�G=C��=OM��X[�R��<n���os�<�^��Ȍ��e�R�I�9
<��伡
d�9�M=�X�<���5Y=�N=}ǣ;�λ*�ν���=@5���F���E=D����	GD���/��]J�$�Q=:��t�����=Z{�<D׽�i4���N��X��X?,���9<֗�<�=�������s!��>#��*����:s�}=��};���=�򽻺����|c>�i�,�M�4=�`>=��^�.�"=F��=r������<[d���缶=E%�=�X4���<�P�<TԻл>�������<F�[=�׬<՞�=��;���=��.>8	o=͘w=n� =�a�=���;�*>�>�@��t�%�$>���<��U�h�=f&�"���'�=!k�}���3½�̛=#�5��[^=d!�=D��=��b<��=k伽�a<�I�=do�(>y����)c�"����H�=���=�
�<ճ�X"<�޼=��E=n�3<�w�:�C=�T�ڕX��i��+�=P5�=2G
EStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp�
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
�
`
'__inference_fc_11_layer_call_fn_2780987

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27809822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
]
A__inference_fc_8_layer_call_and_return_conditional_losses_2780475

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
�
�
__inference_loss_fn_1_2780797:
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
�
`
A__inference_fc_3_layer_call_and_return_conditional_losses_2780467

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
�
_
A__inference_fc_3_layer_call_and_return_conditional_losses_2780522

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
�'
�
 __inference__traced_save_2782542
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
�: :@:@:@@:@:@@:@:@@:@:@ : :  : :	�
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
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:(	$
"
_output_shapes
:@ : 


_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	�
: 

_output_shapes
:
:

_output_shapes
: 
�
a
B__inference_fc_11_layer_call_and_return_conditional_losses_2781179

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
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_2780396:
6fc_6_kernel_regularizer_square_readvariableop_resource
identity��-fc_6/kernel/Regularizer/Square/ReadVariableOp�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_6_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
�
B
&__inference_fc_8_layer_call_fn_2780480

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
A__inference_fc_8_layer_call_and_return_conditional_losses_27804752
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
�
]
A__inference_fc13_layer_call_and_return_conditional_losses_2780433

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������^ :S O
+
_output_shapes
:���������^ 
 
_user_specified_nameinputs
�
a
B__inference_fc_11_layer_call_and_return_conditional_losses_2780982

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
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�t
�
B__inference_mnist_layer_call_and_return_conditional_losses_2781350	
input
fc_1_2716041
fc_1_2716043
fc_2_2716046
fc_2_2716048
fc_5_2716053
fc_5_2716055
fc_6_2716058
fc_6_2716060
fc_9_2716065
fc_9_2716067
fc_10_2716070
fc_10_2716072
output_2716078
output_2716080
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
D__inference_flatten_layer_call_and_return_conditional_losses_27802042
flatten/PartitionedCall�
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_4/ExpandDims/dim�
tf.expand_dims_4/ExpandDims
ExpandDims flatten/PartitionedCall:output:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_4/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall$tf.expand_dims_4/ExpandDims:output:0fc_1_2716041fc_1_2716043*
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
A__inference_fc_1_layer_call_and_return_conditional_losses_27812792
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_2716046fc_2_2716048*
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
A__inference_fc_2_layer_call_and_return_conditional_losses_27807122
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
A__inference_fc_3_layer_call_and_return_conditional_losses_27805222
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27808052
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_2716053fc_5_2716055*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27810322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_2716058fc_6_2716060*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27806202
fc_6/StatefulPartitionedCall�
fc_7/PartitionedCallPartitionedCall%fc_6/StatefulPartitionedCall:output:0*
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
A__inference_fc_7_layer_call_and_return_conditional_losses_27809922
fc_7/PartitionedCall�
fc_8/PartitionedCallPartitionedCallfc_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27804752
fc_8/PartitionedCall�
fc_9/StatefulPartitionedCallStatefulPartitionedCallfc_8/PartitionedCall:output:0fc_9_2716065fc_9_2716067*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_27812282
fc_9/StatefulPartitionedCall�
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_2716070fc_10_2716072*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27806832
fc_10/StatefulPartitionedCall�
fc_11/PartitionedCallPartitionedCall&fc_10/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27812012
fc_11/PartitionedCall�
fc_12/PartitionedCallPartitionedCallfc_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������^ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_12_layer_call_and_return_conditional_losses_27810052
fc_12/PartitionedCall�
fc13/PartitionedCallPartitionedCallfc_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27802152
fc13/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc13/PartitionedCall:output:0output_2716078output_2716080*
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
C__inference_output_layer_call_and_return_conditional_losses_27807802 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_2716041*"
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_2716046*"
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_2716053*"
_output_shapes
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_2716058*"
_output_shapes
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_9_2716065*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_10_2716070*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
�	
�
'__inference_mnist_layer_call_fn_2781447	
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
B__inference_mnist_layer_call_and_return_conditional_losses_27814282
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
A__inference_fc_1_layer_call_and_return_conditional_losses_2781279

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
�
_
&__inference_fc_3_layer_call_fn_2780598

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
A__inference_fc_3_layer_call_and_return_conditional_losses_27805932
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
�
{
&__inference_fc_2_layer_call_fn_2780719

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
A__inference_fc_2_layer_call_and_return_conditional_losses_27807122
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
��
�
B__inference_mnist_layer_call_and_return_conditional_losses_2780960

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
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
tf.expand_dims_4/ExpandDims/dim�
tf.expand_dims_4/ExpandDims
ExpandDimsflatten/Reshape:output:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims_4/ExpandDims�
fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_1/conv1d/ExpandDims/dim�
fc_1/conv1d/ExpandDims
ExpandDims$tf.expand_dims_4/ExpandDims:output:0#fc_1/conv1d/ExpandDims/dim:output:0*
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
:@@*
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
:@@2
fc_5/conv1d/ExpandDims_1�
fc_5/conv1dConv2Dfc_5/conv1d/ExpandDims:output:0!fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_5/conv1d�
fc_5/conv1d/SqueezeSqueezefc_5/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_5/conv1d/Squeeze�
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_5/BiasAdd/ReadVariableOp�
fc_5/BiasAddBiasAddfc_5/conv1d/Squeeze:output:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_5/BiasAddl
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
fc_6/conv1d/ExpandDims�
'fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
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
:@@2
fc_6/conv1d/ExpandDims_1�
fc_6/conv1dConv2Dfc_6/conv1d/ExpandDims:output:0!fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������@*
paddingVALID*
strides
2
fc_6/conv1d�
fc_6/conv1d/SqueezeSqueezefc_6/conv1d:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims

���������2
fc_6/conv1d/Squeeze�
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_6/BiasAdd/ReadVariableOp�
fc_6/BiasAddBiasAddfc_6/conv1d/Squeeze:output:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
fc_6/BiasAddl
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
	fc_6/Reluz
fc_7/IdentityIdentityfc_6/Relu:activations:0*
T0*,
_output_shapes
:����������@2
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
:����������@2
fc_8/ExpandDims�
fc_8/MaxPoolMaxPoolfc_8/ExpandDims:output:0*0
_output_shapes
:����������@*
ksize
*
paddingVALID*
strides
2
fc_8/MaxPool�
fc_8/SqueezeSqueezefc_8/MaxPool:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
fc_9/conv1d/ExpandDims�
'fc_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
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
:@ 2
fc_9/conv1d/ExpandDims_1�
fc_9/conv1dConv2Dfc_9/conv1d/ExpandDims:output:0!fc_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_9/conv1d�
fc_9/conv1d/SqueezeSqueezefc_9/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_9/conv1d/Squeeze�
fc_9/BiasAdd/ReadVariableOpReadVariableOp$fc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_9/BiasAdd/ReadVariableOp�
fc_9/BiasAddBiasAddfc_9/conv1d/Squeeze:output:0#fc_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_9/BiasAddl
	fc_9/ReluRelufc_9/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_10/conv1d/ExpandDims�
(fc_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1fc_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2
fc_10/conv1d/ExpandDims_1�
fc_10/conv1dConv2D fc_10/conv1d/ExpandDims:output:0"fc_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_10/conv1d�
fc_10/conv1d/SqueezeSqueezefc_10/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_10/conv1d/Squeeze�
fc_10/BiasAdd/ReadVariableOpReadVariableOp%fc_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_10/BiasAdd/ReadVariableOp�
fc_10/BiasAddBiasAddfc_10/conv1d/Squeeze:output:0$fc_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_10/BiasAddo

fc_10/ReluRelufc_10/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2

fc_10/Relu}
fc_11/IdentityIdentityfc_10/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_12/ExpandDims�
fc_12/MaxPoolMaxPoolfc_12/ExpandDims:output:0*/
_output_shapes
:���������^ *
ksize
*
paddingVALID*
strides
2
fc_12/MaxPool�
fc_12/SqueezeSqueezefc_12/MaxPool:output:0*
T0*+
_output_shapes
:���������^ *
squeeze_dims
2
fc_12/Squeezei

fc13/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2

fc13/Const�
fc13/ReshapeReshapefc_12/Squeeze:output:0fc13/Const:output:0*
T0*(
_output_shapes
:����������2
fc13/Reshape�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�
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
:@@*
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:@@*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@@2 
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
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
A__inference_fc_9_layer_call_and_return_conditional_losses_2780661

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
:����������@2
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
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02/
-fc_9/kernel/Regularizer/Square/ReadVariableOp�
fc_9/kernel/Regularizer/SquareSquare5fc_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@ 2 
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
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_9/kernel/Regularizer/Square/ReadVariableOp-fc_9/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
B__inference_fc_10_layer_call_and_return_conditional_losses_2780571

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
:���������� 2
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
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
.fc_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype020
.fc_10/kernel/Regularizer/Square/ReadVariableOp�
fc_10/kernel/Regularizer/SquareSquare6fc_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2!
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
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2`
.fc_10/kernel/Regularizer/Square/ReadVariableOp.fc_10/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
`
A__inference_fc_7_layer_call_and_return_conditional_losses_2780639

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
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������@*
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
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_2780581:
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
�
C
'__inference_fc_11_layer_call_fn_2781206

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_27812012
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_2780209

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
D__inference_flatten_layer_call_and_return_conditional_losses_27802042
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
�
|
'__inference_fc_10_layer_call_fn_2780690

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
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_27806832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
C
'__inference_fc_12_layer_call_fn_2781010

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
B__inference_fc_12_layer_call_and_return_conditional_losses_27810052
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
�
`
B__inference_fc_11_layer_call_and_return_conditional_losses_2781201

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2781657

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
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
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_fc_3_layer_call_fn_2780527

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
A__inference_fc_3_layer_call_and_return_conditional_losses_27805222
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
�
B
&__inference_fc13_layer_call_fn_2780220

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc13_layer_call_and_return_conditional_losses_27802152
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������^ :S O
+
_output_shapes
:���������^ 
 
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
!:@@2fc_5/kernel
:@2	fc_5/bias
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
!:@@2fc_6/kernel
:@2	fc_6/bias
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
!:@ 2fc_9/kernel
: 2	fc_9/bias
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
":   2fc_10/kernel
: 2
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
 :	�
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
'__inference_mnist_layer_call_fn_2781447
'__inference_mnist_layer_call_fn_2781627
'__inference_mnist_layer_call_fn_2781466
'__inference_mnist_layer_call_fn_2781646�
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
"__inference__wrapped_model_2781148�
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
B__inference_mnist_layer_call_and_return_conditional_losses_2781537
B__inference_mnist_layer_call_and_return_conditional_losses_2780960
B__inference_mnist_layer_call_and_return_conditional_losses_2780386
B__inference_mnist_layer_call_and_return_conditional_losses_2781350�
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
)__inference_flatten_layer_call_fn_2780209�
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
D__inference_flatten_layer_call_and_return_conditional_losses_2781663�
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
&__inference_fc_1_layer_call_fn_2781357�
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
A__inference_fc_1_layer_call_and_return_conditional_losses_2780455�
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
&__inference_fc_2_layer_call_fn_2780719�
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
A__inference_fc_2_layer_call_and_return_conditional_losses_2780549�
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
&__inference_fc_3_layer_call_fn_2780598
&__inference_fc_3_layer_call_fn_2780527�
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
A__inference_fc_3_layer_call_and_return_conditional_losses_2780467
A__inference_fc_3_layer_call_and_return_conditional_losses_2780225�
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
&__inference_fc_4_layer_call_fn_2780810�
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
A__inference_fc_4_layer_call_and_return_conditional_losses_2780805�
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
&__inference_fc_5_layer_call_fn_2781039�
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
A__inference_fc_5_layer_call_and_return_conditional_losses_2781257�
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
&__inference_fc_6_layer_call_fn_2780627�
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
A__inference_fc_6_layer_call_and_return_conditional_losses_2780502�
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
&__inference_fc_7_layer_call_fn_2780997
&__inference_fc_7_layer_call_fn_2781196�
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
A__inference_fc_7_layer_call_and_return_conditional_losses_2780639
A__inference_fc_7_layer_call_and_return_conditional_losses_2781044�
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
&__inference_fc_8_layer_call_fn_2780480�
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
A__inference_fc_8_layer_call_and_return_conditional_losses_2780475�
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
&__inference_fc_9_layer_call_fn_2781235�
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
A__inference_fc_9_layer_call_and_return_conditional_losses_2780661�
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
'__inference_fc_10_layer_call_fn_2780690�
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
B__inference_fc_10_layer_call_and_return_conditional_losses_2780571�
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
'__inference_fc_11_layer_call_fn_2780987
'__inference_fc_11_layer_call_fn_2781206�
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
B__inference_fc_11_layer_call_and_return_conditional_losses_2780507
B__inference_fc_11_layer_call_and_return_conditional_losses_2781179�
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
'__inference_fc_12_layer_call_fn_2781010�
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
B__inference_fc_12_layer_call_and_return_conditional_losses_2781005�
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
&__inference_fc13_layer_call_fn_2780220�
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
A__inference_fc13_layer_call_and_return_conditional_losses_2780433�
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
(__inference_output_layer_call_fn_2780787�
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
C__inference_output_layer_call_and_return_conditional_losses_2781657�
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
__inference_loss_fn_0_2780581�
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
__inference_loss_fn_1_2780797�
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
__inference_loss_fn_2_2780517�
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
__inference_loss_fn_3_2780396�
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
__inference_loss_fn_4_2780820�
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
__inference_loss_fn_5_2780970�
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
%__inference_signature_wrapper_2782477input"�
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
"__inference__wrapped_model_2781148y!"()9:@AQRXYno6�3
,�)
'�$
input���������
� "/�,
*
output �
output���������
�
A__inference_fc13_layer_call_and_return_conditional_losses_2780433]3�0
)�&
$�!
inputs���������^ 
� "&�#
�
0����������
� z
&__inference_fc13_layer_call_fn_2780220P3�0
)�&
$�!
inputs���������^ 
� "������������
B__inference_fc_10_layer_call_and_return_conditional_losses_2780571fXY4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0���������� 
� �
'__inference_fc_10_layer_call_fn_2780690YXY4�1
*�'
%�"
inputs���������� 
� "����������� �
B__inference_fc_11_layer_call_and_return_conditional_losses_2780507f8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
B__inference_fc_11_layer_call_and_return_conditional_losses_2781179f8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
'__inference_fc_11_layer_call_fn_2780987Y8�5
.�+
%�"
inputs���������� 
p
� "����������� �
'__inference_fc_11_layer_call_fn_2781206Y8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
B__inference_fc_12_layer_call_and_return_conditional_losses_2781005�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
'__inference_fc_12_layer_call_fn_2781010wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_1_layer_call_and_return_conditional_losses_2780455f!"4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������@
� �
&__inference_fc_1_layer_call_fn_2781357Y!"4�1
*�'
%�"
inputs����������
� "�����������@�
A__inference_fc_2_layer_call_and_return_conditional_losses_2780549f()4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
&__inference_fc_2_layer_call_fn_2780719Y()4�1
*�'
%�"
inputs����������@
� "�����������@�
A__inference_fc_3_layer_call_and_return_conditional_losses_2780225f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
A__inference_fc_3_layer_call_and_return_conditional_losses_2780467f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
&__inference_fc_3_layer_call_fn_2780527Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
&__inference_fc_3_layer_call_fn_2780598Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
A__inference_fc_4_layer_call_and_return_conditional_losses_2780805�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_fc_4_layer_call_fn_2780810wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_5_layer_call_and_return_conditional_losses_2781257f9:4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
&__inference_fc_5_layer_call_fn_2781039Y9:4�1
*�'
%�"
inputs����������@
� "�����������@�
A__inference_fc_6_layer_call_and_return_conditional_losses_2780502f@A4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
&__inference_fc_6_layer_call_fn_2780627Y@A4�1
*�'
%�"
inputs����������@
� "�����������@�
A__inference_fc_7_layer_call_and_return_conditional_losses_2780639f8�5
.�+
%�"
inputs����������@
p
� "*�'
 �
0����������@
� �
A__inference_fc_7_layer_call_and_return_conditional_losses_2781044f8�5
.�+
%�"
inputs����������@
p 
� "*�'
 �
0����������@
� �
&__inference_fc_7_layer_call_fn_2780997Y8�5
.�+
%�"
inputs����������@
p 
� "�����������@�
&__inference_fc_7_layer_call_fn_2781196Y8�5
.�+
%�"
inputs����������@
p
� "�����������@�
A__inference_fc_8_layer_call_and_return_conditional_losses_2780475�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_fc_8_layer_call_fn_2780480wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_9_layer_call_and_return_conditional_losses_2780661fQR4�1
*�'
%�"
inputs����������@
� "*�'
 �
0���������� 
� �
&__inference_fc_9_layer_call_fn_2781235YQR4�1
*�'
%�"
inputs����������@
� "����������� �
D__inference_flatten_layer_call_and_return_conditional_losses_2781663a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
)__inference_flatten_layer_call_fn_2780209T7�4
-�*
(�%
inputs���������
� "�����������<
__inference_loss_fn_0_2780581!�

� 
� "� <
__inference_loss_fn_1_2780797(�

� 
� "� <
__inference_loss_fn_2_27805179�

� 
� "� <
__inference_loss_fn_3_2780396@�

� 
� "� <
__inference_loss_fn_4_2780820Q�

� 
� "� <
__inference_loss_fn_5_2780970X�

� 
� "� �
B__inference_mnist_layer_call_and_return_conditional_losses_2780386x!"()9:@AQRXYno?�<
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
B__inference_mnist_layer_call_and_return_conditional_losses_2780960x!"()9:@AQRXYno?�<
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
B__inference_mnist_layer_call_and_return_conditional_losses_2781350w!"()9:@AQRXYno>�;
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
B__inference_mnist_layer_call_and_return_conditional_losses_2781537w!"()9:@AQRXYno>�;
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
'__inference_mnist_layer_call_fn_2781447j!"()9:@AQRXYno>�;
4�1
'�$
input���������
p

 
� "����������
�
'__inference_mnist_layer_call_fn_2781466k!"()9:@AQRXYno?�<
5�2
(�%
inputs���������
p

 
� "����������
�
'__inference_mnist_layer_call_fn_2781627j!"()9:@AQRXYno>�;
4�1
'�$
input���������
p 

 
� "����������
�
'__inference_mnist_layer_call_fn_2781646k!"()9:@AQRXYno?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
C__inference_output_layer_call_and_return_conditional_losses_2781657]no0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� |
(__inference_output_layer_call_fn_2780787Pno0�-
&�#
!�
inputs����������
� "����������
�
%__inference_signature_wrapper_2782477r?�<
� 
5�2
0
input'�$
input���������"/�,
*
output �
output���������
