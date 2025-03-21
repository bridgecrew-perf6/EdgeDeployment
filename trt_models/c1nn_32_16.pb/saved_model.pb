��
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
 �"serve*2.4.12unknown8��
v
fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namefc_1/kernel
o
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel*"
_output_shapes
: *
dtype0
j
	fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	fc_1/bias
c
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes
: *
dtype0
v
fc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namefc_2/kernel
o
fc_2/kernel/Read/ReadVariableOpReadVariableOpfc_2/kernel*"
_output_shapes
:  *
dtype0
j
	fc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	fc_2/bias
c
fc_2/bias/Read/ReadVariableOpReadVariableOp	fc_2/bias*
_output_shapes
: *
dtype0
v
fc_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namefc_5/kernel
o
fc_5/kernel/Read/ReadVariableOpReadVariableOpfc_5/kernel*"
_output_shapes
: *
dtype0
j
	fc_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	fc_5/bias
c
fc_5/bias/Read/ReadVariableOpReadVariableOp	fc_5/bias*
_output_shapes
:*
dtype0
v
fc_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefc_6/kernel
o
fc_6/kernel/Read/ReadVariableOpReadVariableOpfc_6/kernel*"
_output_shapes
:*
dtype0
j
	fc_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	fc_6/bias
c
fc_6/bias/Read/ReadVariableOpReadVariableOp	fc_6/bias*
_output_shapes
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�
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
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�0
value�/B�/ B�/
�
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
layer-11
layer_with_weights-4
layer-12
regularization_losses
trainable_variables
	variables
	keras_api

signatures
#_self_saveable_object_factories
trt_engine_resources
%
#_self_saveable_object_factories
w
regularization_losses
trainable_variables
	variables
	keras_api
#_self_saveable_object_factories
4
	keras_api
#_self_saveable_object_factories
�

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
##_self_saveable_object_factories
�

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
#*_self_saveable_object_factories
w
+regularization_losses
,trainable_variables
-	variables
.	keras_api
#/_self_saveable_object_factories
w
0regularization_losses
1trainable_variables
2	variables
3	keras_api
#4_self_saveable_object_factories
�

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
#;_self_saveable_object_factories
�

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
#B_self_saveable_object_factories
w
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
#G_self_saveable_object_factories
w
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
#L_self_saveable_object_factories
w
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
#Q_self_saveable_object_factories
�

Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
#X_self_saveable_object_factories
 
F
0
1
$2
%3
54
65
<6
=7
R8
S9
F
0
1
$2
%3
54
65
<6
=7
R8
S9
�
Ynon_trainable_variables
regularization_losses
Zlayer_metrics
[metrics
\layer_regularization_losses
trainable_variables

]layers
	variables
#^_self_saveable_object_factories
 
 
 
 
 
 
 
�
_non_trainable_variables
regularization_losses
`layer_metrics
ametrics
blayer_regularization_losses
trainable_variables

clayers
	variables
#d_self_saveable_object_factories
 
%
#e_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
fnon_trainable_variables
regularization_losses
glayer_metrics
hmetrics
ilayer_regularization_losses
 trainable_variables

jlayers
!	variables
#k_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
�
lnon_trainable_variables
&regularization_losses
mlayer_metrics
nmetrics
olayer_regularization_losses
'trainable_variables

players
(	variables
#q_self_saveable_object_factories
 
 
 
 
�
rnon_trainable_variables
+regularization_losses
slayer_metrics
tmetrics
ulayer_regularization_losses
,trainable_variables

vlayers
-	variables
#w_self_saveable_object_factories
 
 
 
 
�
xnon_trainable_variables
0regularization_losses
ylayer_metrics
zmetrics
{layer_regularization_losses
1trainable_variables

|layers
2	variables
#}_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
�
~non_trainable_variables
7regularization_losses
layer_metrics
�metrics
 �layer_regularization_losses
8trainable_variables
�layers
9	variables
$�_self_saveable_object_factories
 
WU
VARIABLE_VALUEfc_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
�
�non_trainable_variables
>regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
?trainable_variables
�layers
@	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
Cregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Dtrainable_variables
�layers
E	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
Hregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Itrainable_variables
�layers
J	variables
$�_self_saveable_object_factories
 
 
 
 
�
�non_trainable_variables
Mregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Ntrainable_variables
�layers
O	variables
$�_self_saveable_object_factories
 
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
�
�non_trainable_variables
Tregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Utrainable_variables
�layers
V	variables
$�_self_saveable_object_factories
 
 
 
 
 
^
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
%__inference_signature_wrapper_2760358
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenamefc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOpfc_2/kernel/Read/ReadVariableOpfc_2/bias/Read/ReadVariableOpfc_5/kernel/Read/ReadVariableOpfc_5/bias/Read/ReadVariableOpfc_6/kernel/Read/ReadVariableOpfc_6/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpConst*
Tin
2*
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
 __inference__traced_save_2760411
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc_1/kernel	fc_1/biasfc_2/kernel	fc_2/biasfc_5/kernel	fc_5/biasfc_6/kernel	fc_6/biasoutput/kerneloutput/bias*
Tin
2*
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
#__inference__traced_restore_2760451��
��
5
__inference_pruned_2760351	
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
;StatefulPartitionedCall/mnist/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;StatefulPartitionedCall/mnist/tf.expand_dims/ExpandDims/dim�
7StatefulPartitionedCall/mnist/tf.expand_dims/ExpandDims
ExpandDims6StatefulPartitionedCall/mnist/flatten/Reshape:output:0DStatefulPartitionedCall/mnist/tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������29
7StatefulPartitionedCall/mnist/tf.expand_dims/ExpandDims�
8StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2:
8StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dim�
4StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims
ExpandDims@StatefulPartitionedCall/mnist/tf.expand_dims/ExpandDims:output:0AStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims/dim:output:0*
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
OStatefulPartitionedCall/mnist/fc_1/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�
6StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1Const*&
_output_shapes
: *
dtype0*�
value�B� *�4�>9Oǻ�\Q=Eo<�0x�9>�ܼOX��H��=�5�=����}A>�
���Yj���l>16>����.�^���;Q?̼g���s㦼����jn=��>���=Oq�3>.�g>�3>���Yk=�4�=��S��lm=9�*>�qF>���=���X�U�`S�=fX>2�=˴>�&F�_�P�[��<��>Lֻ�0Ļ��8> �3�v����[#����=��>��>���=-<G��=OF=���=X��`>����|@���>>��=��<������[[>����=	;;xo�=S.�<{�t2�o�����<�-��9�>��M��j�;�y>�pL���>亏=���Vj>�ϰ�P�F<�2Z�H	>E|%<,��=28
6StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_1/conv1dConv2DSStatefulPartitionedCall/mnist/fc_1/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:��������� �*
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
:���������� 2S
QStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_1/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_1/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_1/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOpConst*
_output_shapes
: *
dtype0*�
value�B� *�r���    �9R�v;/>�=f;        �J�;�V=��3:Ex�9    �^�=Ev��h�    ?�^�Aʢ�ҿ���<=    !��[��:�Xx����;[�`9�F�;z�Y+ʼ�S�92;
9StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_1/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_1/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:���������� 2,
*StatefulPartitionedCall/mnist/fc_1/BiasAdd�
'StatefulPartitionedCall/mnist/fc_1/ReluRelu3StatefulPartitionedCall/mnist/fc_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2)
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
:���������� 26
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
:��������� �2Q
OStatefulPartitionedCall/mnist/fc_2/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�a
6StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1Const*&
_output_shapes
:  *
dtype0*�`
value�`B�`  *�`~�S�ľ�c��;=���=�Hǽz4r=ac��\>�=c.��9;%��<gԎ��$�9(����֝<ن��G��lֻ��>��=XÞ�K��:!>T[̻��=��f;ez��5�	����3�K�D����|�C0
�=�Y<2���H��<���=�n#�4�:=�E�<���V��[�����<�.�����9W���V���P;�b�=����+�@=u\���5$="�����<���;�,���|�<_U�9�����<��,=ۧ�����<���<X�!<>�9�����P>��=$+���d$>Y��<F��;��C<ZgF�@�����=Ċ/�zwN=l��=*<�W�v�r=�5��_��=�<�='�<�t�=�9�&�<�q�( ���`�k�=U�=�zU>o�<L>I��=�l>Y�=�ZZ<H�=E�y<y<=��=8���*=�[�=.z�=t�^>`�0>�
<;�Ǽ��J=:g����ͼI|a=@��2��<y?Z�E$=4 <�>6�<Kn.=?�=���<�v�=��>*z>ELU=Ha>���<��V<V+�=hؿ;�L�=Q=�Y�=���8�=TR�=!դ=h��=�� >7d9��='}�=�4�=�=�,��Pᔽ�:K=b͞=<Ƿ=�y%=V����Ԟ���>˞!>�_�K+!>�$4�?=>=~��[~���ݐ=Q!~=ﺼ'Î=(�aNԼ� �=j���nk<C�>��=c@��&>�μ�V���;��7��M�]ؾ���Φ ������":W�=���<���a�Ľ�=r���4"%=}s�<`h�<�뗽��=A&=�g�<N�E��t�<{���ټ}�o=:\���M=�X=�!�<?w�<��Ž�; ��w�=�W�;���<땀=�P��OX=�g༠g���B"=0/��,e=�=4*�=��Ƽ��;�!<�t�<���=�)�=�3���g�ׇ�=�d4=f=� ��+F�=K����߻�����x��ν���.E=�J����<�@�YE��X́���/���=� >,<�J�=��!��=����5W=D+>Xд� ~;���=|�J�\	��m���x��	#�.B>M�P=y��=[�=ɠ���p�<��
>�ҏ<�2��o�;����~�"m	����<���=����=$�=�7�=���;JMh>�?=�iV��ٟ<y�=��ν_�=��"=��Q=`�=���:JZ=�I�=0N >.�;�*=@ڭ=��F�}��=����k>�)�z�\�(>;�Z=T+����>�=\7�<��T>��0> >��=�>#>�	�=�p�p{>�S�xuC��B>��->��<�<>N,=�C[>Q�>�$>/�=��%>��=z�n=+�>�ZQ�\�">���=�z=�4�=���0Eh��1�.�=�m0>�,2<���=�ۆ���>����ս:�=GW%��>*<k��=�`񽍆���ԅ<�A��&7��Q>D��=��=K�=���<�W�=��=*�I��=��E�0�[:C�5�a�G�=W^�=]M7<[�
�x	O�q?8=�f�=�]�=H&���{�G����4��I=/ v=Ɉ�<H�u��6�0kM=F =����:�X��/��x9 =�o�<��=p��<7y�b��=��<W��;/�
�t�_�=C��=��2=&��'OL=A�l�rFA=Ǯ�������=������h<{Ly��^���I>��˽���2�
>���=�e�})����G=����A�����A)����=��� �=��;a��=��k=����!j��+�=� >\�9=mD�=}�e��Ɔ>��>�4<s�=Pv,;����?\>� ���̉=�@�=$Rs��\�	==��:>�i��>���<�2�>̏>#;?�<9%A<'���4;��:�o.�����L^o={
>?$'��^=N�=���]>P��<�����kp�m�ռ�Q����=�`�^���l�=U]o�(�=Ƀ�=��=��m<���<��ü~#�=�p=�#��򺾛�9�g��D =	�>Z�l=7E�����;�<��H=��=��<�b>�˨���-^=�!��C�=|��RT]<%�l��>���<(�ý_��<�I��"}<�%�=���'�=��^�"�%=M��=�S���dü�սm5�=�4�>��;��:��Q>�}�>�~�>]g=�#�=f'�>�[=�-�=�V<W�q>�|�=�`��c@�;=E�����>:��>��R>��6=��=@��=��>��>yI�ֳ ?ȑ�>ꎜ��8�=<P"�W%+��s ��5�����>M�[>�N4= Cʼu��>Ҡh> B >��K=��>����<��s=�ӽ0)=WnS>�j�� �>��>Ƅ�>����O�=��/��Vj>�>T΅=���<��.��	��s���r;�Љ=��<�N<=l�	<xF:��O�c�^;�&ս`�==Dݼ�1���:<v�˽yI)=��e=id��W>�=ۊa��=<z�\;�/=�.��ܙ�=~����;���;�-=�Ƨ�^����k=���=�=sl>-��=��=9� ��0>�H����=����d�B>yN���}�;�̦���o>�;���=ҕ��B�>N=Y'�=�����,;=�c;��=��=��>8�=��!<D֍>���=�վ=D[�=�0���T��-=��1:#��5<�R�#�<[R<;�=����g˽�|����
���<(x��~=��һ���<���FF2=A��<�� <�;����~���_;}㮼���<MJd<ЅW��ɰ=#��=qҔ<}+>���<� �=2]>K�T>�>\��M>��=��=���=��ͽX�=O;�=}��}��=��<��黙3j>�n>�I�=k�`=�N�<���=Z6�=��>xPm�*�>5��=��6q=�j;���R{߽���=u�=XF�q�%>�OA�둞=c,�=,�=u}�=��B={��*_Ӻ
:�|�Ƚ��+>�潽^ =�p
>X��=�Ž�[�=�m�<�-;O��=?r�=7 �ЛH�h䷻��ûվ�
ٻ� �
 =�I)>�=B�P=+�?�x��>kG>d��<&�P���D��� �=�����导�&�=[; =J�&>J�<ZP�>泮�n�>�*-=vm>�Fc<����{<�O!=�y���ƽ��?=_Di�Cｅ��=�v3>ux���=�\-�ki>�݇<���=]�!>��������=�$���!8�-	k=�0,�hy�=��>Yr���sY�N�b=;�=���=�5�=���<E�=�v��MTǽ��սW�����<���ș��{T�ݰ���C= �<�\������=�U!�
r)=/������`�<��=@A��S��[m����=5�A=��=��<�^�<�^���@=Z��;�M=���<�K=�f;k�<[Er�ot]�-�<L��=`��=%=D=^ҽ5�q>��<aq;���*��Z�=\���b��=�5�$�(<M��=����W��<"�x=ڈ�=8L���=���=Y|���=����S��Q���-��^��*ӽ@�����׾�a�<2:>n =�;>ؚ����h>l	C>�O���{�=w`N�lW&��8�=k*W�(�N�>ؘ���y=Uw>v,>���<�:x>��<��d>
N�= ��;�*y=� _�Ob��M����?���׽O�U�NC�<{n>�e=�8,=�Z��0��=���=���s�>w�<6�R��=��G��l�=F+�=�"���jU���@>�/s=�	^�]:�=�a�<c4=="�=A8=�5w<P��X#�y��-�A=߹��5t��ա=�b���0r=�X�s���q<� =Oa<��񼠽a����Ie�=����=<?���@�����`�"��Ђ�TE�<�w=x���I�2�_��C�=]>�����g��il��;b��A�i
���=��=n즽��?�w���x�=#�%=����I�=�	<1�ӻt�=�*��P�<3�=y�M��_<-O=���<�3���=u-=�G�"9�=:��?]$�x�<��)���[���.;�C�=k�� �>� 2=L��\m>�D5�\�>�A<��0=��=�G��v�=r��=�>�=`j����=��r�\�>eӭ=���=n�=w��= 0�<֡�=���= ���i"<M�<R�<�咼��<�6-=��s;�6�����|X�<�t�,E�=�e��;0�u��-<X���ԇ �Ls�<s�!=�-�<�0������>��ϙ=�q�v��<Gzؼ �׽��ֽ��C;c�c���<H<',8�����<�n�=�W#;=*�=��=��ν��=
[J�m⯽r�7=��.�x��=��弫�=f�=&�~=��<;�����<i����=�=�<�ﯼ�v�=n�<��="k�=��<�_�=�x=�X�=�Tv=�r�<��=�"=�\�=:�<��%>-�>�>8dk���>U~���Ȇ<�F6=?��<M�=�t=nǄ�,O���;=>��=�B>6�Ƚ�7_=�I=�Ѱ:s�>Q�
>c�=~0S>��;>�R8=a�>u�P�<��>m��<���>�]>[{�;���<L��=��v;�D=��5=�؈=*Ґ=��==��;��D<���=��=5�*=|��=ICԽ�$�=G��=��:<}��=�1|<	�.>�.*>��>�>p�Ǻg=yP�#�ɼ�Q3>�}W>��=f^�=h�>�K>+!Z���f=s�e=�ڪ��P�=��<���<�9>�;K���>��=<��=D�ͼ(��=�S=K�>0��=Od6=9��=�~���/�<���W!�<���̻{�<��(=���;7���+�E<�m�<��G�X�^=4ѷ<�鎼Br�+���ͨ��<��8��֑<���s�x���������@<�Љ;��f<?˽��؞����;�_S=�)=_<����nH���->,w=�:�=I�<﷣�:!�<N3�<n��@�V�7>F=�rV=%��<��|=M&�=�Ii=�J�Ѓ��Z�=g�=��<�=�?��=�uμB�W�fu="<�<}�	����<p��;B�A�2��s�<Θ$=�'�=]�=Y�����>m6�<r8M=��=�D����=TՕ<I�o�8��=��i�3����f�=1�=|=���=pRR�ck8���=�:�;=�=�Y�=����7D�=���=f��;̬d=q�=9G��<�=�=W�>��=�!�=@(=�=�p����<�G=W~ϼB�=N�$=�Ut�&j-����<�B�=�n|=�>�#>�绥��=���=w�<F�p=0�Ƚ�>3o�=B`�<�%,>��=\9�=���,�=n���;���	>��R;������>w��<���=�b�)
�=3��=O�=<�IW�$�j�K��=�
=s�5>R�����A=Kx5=m:�=��X=�dE>D��<�>�# >:��=ŨB=��켴N8=��=��=)��=��<Y�>k�=r�>P��=[�}���(>ꝇ�>\�<0y>���=�G��:��<�T�=�L�=<�/>Z�=��<Iʂ=7g�=�Q�=c�L=N�n��.��l�K��yc=~�л�«<��X'�<]ǻ�|��Ҍ�#�J������3=�x��J���g:;+*;���H��ϙ<=>�<��@������K%=uU=]Ǖ=E ���Ƕ�{�<k���d���9'=��޻�_=:�<\��E�<�^�=3����2�%#�=&�ͽ�_=g{n��!�%��=��t��/ļ��]����=7l�:H��?�=]>�菽�&����.=Q��c߼D�vu�Z�(=�"�����t0�ϏO=��<��;k����=��>�2�>���=��߻$$?�_>/<?�2>�WB=���� ^9>�T�;�����>6�=RF�>��>���>%���g>s��=b\>(�=M��=E\�=� H>D��;�D���<�X	==��<c��=�P >�q�=��/>wc�=�>���=�n$;{��=���<�\<�">U)��0=="}=mW�<��=.�+>��T=���<�ѐ<,��=�s:=l=k ~=�i�=.1	<6�X=���={��B�<CR�;
֓�+)�=5�e=U�뼧�����<��t<��i��"-=o#�����<��=�>�=aĚ=sF�=ou��}G���'=��]��ӥ=���<^�.;�鎼�����3U=�͕<K��h�B=��u<�n�;����,=

ٽ���ZX>�����M�Ț�=9-�nI&>�;ƥ>B��=$�E��#��u�=�xڼ���N5�=���k@= ��=K�=��U>(%O>�a����>�u�>7�&���b7�޽�<��w��\>��?b��>��>�Lｼ-?�`?}\�=!�>�HF�`>̞>Vm+�-��=$�?a�Խ��>�ͫ>Q��>��ܻn[>� �>M�?U½>�4����>�jD>��������>T[�<�,�钻<.�^�v����=�a�<�y����<g1s=���<�B4<�s`�3��<�=A��x!=0!��5`�7Ik=���<aH����/=c� =\w2=�$�����<i�ɽ�����==Y��<�w��4%��㈽B����1�=0|����j2�ot�=�!�wv= �ͽ�oe��=�;)E[>�ۋ��7Q��ZQ=Bd:���ƽ���������=��r<D>!=$=Zѫ=0Wa={B<^��c�=�m<�=�n�<�q�)�5�Zhq=�4�6ʻ��)�?�<g��=� �e����8�XU����n<�o�<��=�E���|�c��=�$c<�=Iq=�=�-`=H��;��=ۜ�=4!�c8�BJs=F��=PwL��W=��Y���=�9��*)��9�> �������1�=����O��=
�<�E>x[j< �C�h雼�U�>~ٽ�ꆾ�r>߹ʽ��Y��C�=�9�= j=�]B>�����KZ>i��=,w�<�$�<%X��� �=[�O=ׯ�;ɩE=C�$=^[;>�0�硥�n�*>�l<v^�=�5�=�v�=���=�=A���l�=�ȉ����=Y�=J������<���=.�=�i�=��[>�����>�'���J�<��=5>���>���=0(R<�H�>Tr�>,��=R��=��>��N>�H����9<0r_�����c >!��<�x���ֵ>�2{<ٚ>L�>��>(�<��=F�R><!�>���=�=~�=]��=K�PuԼ�d��W�=`O�<0k>��X=�1���>șN��� <x���~���>�:=�X�=rn�=O=������=�و��曼�$ =3�t=��B�Q<ˤ�=�5=lV�=&=��N>�=�)�<�T�=�� =y]d=���뮉���0�ݤ>K�<����AȻ��ҽ-��R)<0��O}ϻ]-[�d0?����w��;�)��^�����F3=(V�#'<]I����<�C�p��=c�mĩ�Z#�=��;��ì�ښ�=�=�=�vi=��<�\�=�D�=G��=?��=�s�;��G_>(nx��ǎ=��I>�ŝ<�FX���>^1�=~=>���=�<� =桀=�MT=䜾=pc�=��2;vv�<���<�=c�<���W���hؽ �=�'}>���=�
>������>�hL>g�S�m��=ѣ�����<�>V8���������>���:Jޛ>�k�=��>p�O�:P�=K��Km�>�]S=%�K���>�=5~Z�Ѩ���G���#=y9:=a�=�<l�����= ߏ=}�=3j�<�9M��� >8G�S�s=~a�=̕��X¼��N���нȓ��J�=���=;)=^��=��=�i=�@:u=��=�D=O��������)��i��l���k=��5��+�M���#3���0���Z��v���=��s��S�=�`P��{�<�o:=Vb�<�a����=\�����S=@Z�;g[�o������;h�l�0��A4���*�<Ʀ���k���O=m���E=�]R��V�=�3
>LǼ��=�=��Q��Q<=�:7��?='&=��!=�Ro=�*=�=��w�	=���=�
K����=A�<5"$=�抽���=l�0>K�`>-
3�a�j>���=���=K0>�<�։>����=g��=��}<.�=���=��"=���=+���>������3>��=e0�<�,=L�f<c��=->L���>�蔼q��;��E=7�=~�}=�d>F�]J#>�v;=ҙ�=�=i=�H���k�=�O8<�s=R��;]�R=W5=�Z��h��w�q=�a�����<�:Wt=8r<w2%<S),<�A,=� Q=(y���� ��p����<�YܻyW�� �o�mw�;�4���O��Fý`q�����$~=Kl/�����0�>.���/�#�m�>R�_�m����+=T|��u�=�>���>%	�<3m�O�=��b�ߨ�����=w�����=�"�=1^�=޺��^��=	����.�=�P�=3[�4�8��%B=X�#�֑�����s���JV� >�=�2��>�. }��
����i=��*=��=��h����ʥ^=��Ƚ��	@���C>�q�]�,=�2Z�Z�=5�׼�=�}�<z$S>T=)e2: ��m��<E��<���=��=֔����<��*��t>�砼����v���G#E=*�	=���=�`��Sj=W=�~��E�=M��=���=yՕ<�p�\�<��s>�	=����ɪ��!�=�v>C�=�q�;���;�O1>�y��՜�=A��<��b>L�Z=e�=��;)�	> �O�nJ@=&�=R�L=�Y>�+=U�7����=W�==:�=B�ػt��<#ð�y?�=��=���=�r>>���=�Ru>�� =6$w��C�=x�!���>g�	��v�� ��im��H�2$л�=  ����=����wj��N�=鯊=5����.<����ɼנ:��E�p�;���0�<ף�o}W=�Z������<��%�[kB��ӛ���7=D"='6���Ö<�Ľ���<PQf��L���=@Aغ8�d������+�<OpU����V���gd=G��:`��"�T<����ξ�^�������*~��]����<�x����}�}��8 h=/�=��=9��9���@��=cZ��U�_��>ѽ�~����=q��H�
=�ᇽIh�=�{�=n�'�b$��u �=�B*���н�}�==���=&��=='���p���=���<���=T�T=�ݽRU�^'
����=��ϼ��=Cz
=�̀<v�=��=�.��'>B)�<?1C=�3�e=�U<�I�=B�a<E�r<|�e=`�����_>g�O=��'���g=OW>�
^=��<7���i��=o�=ʑ�<�b> 6=Pc&���Ȼth�=8���;��>:�$��彦~ �篎��������z+X>�M[;�u��j5�=�冻Hx��(�����=Q����=A/�=P�F��>I<ň�<����t�=fS=v�߽k����R���M;=�:�<���=똉��-����=#������=R]E�g>8�c����=2�>�����a~��k(=mE�;ko�&>>�S��R@����=��>xա=�n�=�$��A�[>�ێ=K3۽7�K<F�=(<����j�6�/b�;ZB�<�"��I<�z<~OL�~X
=o�9��Y�8.�]�k�[=��q��)6�á�=�K6<���ә&=&��z��<e����x	����	��̹֟�Ҩ�TM��5�la��%=%�#>�����c���(;�Vнz��=���=U�޽5m&�s~��@ =��<����=�}���l6��NM=��#>��%�F����舻�ѓ='��;T9��^2Ž��:���@��E�=v=.	*=��#��FF>S��<*,>lK�>�?��>��=�̒>� ?L�<%W�>jm=q��=�i>��=r�Z����>�#ɼ�v4>���=�U�>�pf<d�>�T><�>��>o�@=X��>���>�Cz�C�=2;�k�=�[==o6�=۽X;<?]S>�ɱ=ZZ">p1O��S�=�+<���=�
>��!�B�:�༇�Ǽqý�"<�����}�j)=[�=�m>�=���>�I=�%ҽ��N�[��"�t��ʅ�ᑛ�[���Ԗ���&=\��ȿo=��z�p�Ҙ<=2G�$�=���=��ż�>�=]�_���Ľ+�'��N=�+�=��<��8<$Ӝ�L�<
Zp=m1���O]�=U[O�0�ݵ�B�Ǿ�n����>��i���"sv�!�t�iP5�B�J��q��@ =���<�k�=T[�<})����'�ؚ�=��o���h��D�=�ֽE�a<�h=��=T�>�뼇f�<x�=
>-���C��M��=�Be>��2��ѵ:�[>w�>��e<�<���=L�?�L�x�D>򃌼��>��>��ּ�Ͳ=薪>{ѱ��� >삣:!>n>V뇽 �,>���>IK�> ��>��&<+	�>
	�=���/��=ƞ�����;ڼ,J�<a��<��&��ƍ;Ê�<����!��ft�l��=H��)ܼ��Z����<�ؼ�;��l�� �=�ἂ3=	�Ž;1����;y�ռ���<�+�=`@�<b-�BiP�l��<�z��r�$��<�>�<�T�@�>b��K�B=W���3`�Y���>�<����=�Z<Z�(�`<t/�H�����	���*�`��=g�x=i_�=��s=��=�^R���<7���l*�=�B��.\���=6��``�<��O����H�/��=*�= �<b�<�>��=����;zж��zl�qJ�%N�<>&���>.�ɼ�PW=�5=�{8=τ�<�	\=4'��De�<(�G=Z#x=S��8�<�F6=l?V��n`��=��.�}���O�p=�8t���Z���SP���k�<�6=>���D��SR;W�/<�*���@���r�=�����
����*�>\g8���;[�x�c�=�� =os��5f,�B�X��퓽����ǵ�=���7��r=�{H�ֵ�Fwl="��0F>8����>�uf<m���2.����.�*��h���Y>����?�8�����⽽ny<]˭=�g���&>d7>�c���wm��*�;4�>QWm=4�=Z��>�]�>p������X$	>jZ�>��/� x=zo��nf�=���>�X�=�t��1�>qdn=i�w>|L���>`1�62�=�H�=���>8�>���=��>�^�>���=��d<Z{7:&�O�v�޽��9=߫';���U�=n�����@�A��@J��++=ㅹ�\4�=��=���T�<�H��n��iQ��7=>�G�����K����=^�=j�
>M��<t�&=�:�=�c�cWw�0U�����"����<=
����0��;�I�<�s���ҧ=�<�}�D-�����(�=*,��/�<�ˬ=�+=$;�;Gة:��0��D�;?),<J�>���=�I�<�}�j�=Ě8=��<��";��|;"<=4[�;h:�<�)7�Y�z=�`�=�_==��s(=	P�<q>/�����,>��>�����r<�
�=��z��c�<��>�
h��S=%�p=-�1<9��<�>�1=�Ǭ=Җ>xٲ�g�=4m�=��.>�ˬ�E,>���>���>W�-=,j��H4>#��>$�U�>}�=q�>�jP>�7�a-��]j�>��H=��V>��=�_�>-U=ۯ�=_�d>j��>6�v>�S�={��>�>w�\q����G����;т���=]���9��4��=k��".׼"z=�)=x#=��6.v>�s�=�e"�x�7����<�!޽�+/��0=){н����j$�=	Y�=g-=@��=N�n<I�V;b�=����m"����`=��<��E;�'���\>2=4��=��ƻOϩ=P�<�C �M�I�*�5=k���F<��<�-���ϼ�4�ü@7��x�8<��ؽU�>=V�e���G<�t-�x��}����D/<s�=�H<�����-���I�Z��=���<$Ok�(�b��W\��m��26o����;��=L5]�dҽ=Ze�=��5���L=��<t�������=2k�/Q����>�j�=���=�>l���_��=ޒ�<v�ꝙ�28
6StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_2/conv1dConv2DSStatefulPartitionedCall/mnist/fc_2/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:��������� �*
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
:���������� 2S
QStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_2/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_2/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_2/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOpConst*
_output_shapes
: *
dtype0*�
value�B� *��b�`��=��=�@�<��<yG�=q��;;p�=�� <Ǖ;�������=��=���<{A�=�ջ�u�<%��=�Ԓ=�X ���~=\Ļ��N=,h�=���<�u��#@���|����=��=�D=2;
9StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_2/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_2/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:���������� 2,
*StatefulPartitionedCall/mnist/fc_2/BiasAdd�
'StatefulPartitionedCall/mnist/fc_2/ReluRelu3StatefulPartitionedCall/mnist/fc_2/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2)
'StatefulPartitionedCall/mnist/fc_2/Relu�
+StatefulPartitionedCall/mnist/fc_3/IdentityIdentity5StatefulPartitionedCall/mnist/fc_2/Relu:activations:0*
T0*,
_output_shapes
:���������� 2-
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
:���������� 2/
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
:��������� �2R
PStatefulPartitionedCall/mnist/fc_4/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_4/MaxPoolMaxPoolTStatefulPartitionedCall/mnist/fc_4/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*0
_output_shapes
:��������� �*
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
:���������� 2T
RStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_4/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_4/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:���������� *
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
:���������� 26
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
:��������� �2Q
OStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�1
6StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1Const*&
_output_shapes
: *
dtype0*�0
value�0B�0 *�0�M�ʷ=��g=�嵼jf^�r�@��+׻��~�3�y���ļ�S����ռC蚽�(;=:��Wt;�e�ý�h���%<};׽�8�<\!���Ư=$�C�K�>ʽ�=Ф=eځ�X���=1`�<��J>�*n�\�=NC�<�s	�'=,=��&���=�S=d\>2�]=̠{��������<�E� �t��t1> P<jj=v�[=�G�=��/=O�$>v� ����=�s�1�9�|�=բ:=���=��=���;� 2=٢Y=��3=�v	��<Wp��h}���:>�R�=��5�g����=Qz��b@���>��;��I�e
�;�����
=�D����(�0W�/D>�[:���`>8=���=�y;<@���jݝ<����G<>h+���ѿ=���=���=e�<�	�7>�hH�iο=�B��2�J��k>�X=lp�=_�>���<���P���*��d�J	#�\�U=�>,=w��=��=�j>==�W�򺠻i�!��S:���<˘>���~9�=ݢ	=�j�=`����t�� �>�h�� ��//�=��=�o=KQ|<s�>mx�z��'��=�9c=���L�!>P?k�G�=Ɩ��L�]��k��y��=�4n������>i�(�;���J���;Q=�7�B�{<em >���=}���k��v���=ڭ��cƼ��g��hR<��~���<jA*<d�=؂�����=_�Ľ
e=׀K�$Q7=����C����;�fT=2�=��8>�g���_�ݫm=3��<����'+��	�<�F��'�=R/=����;gH>��+}�m��=%o=|�:*m������>�~���,�=B�';cm>�$���A>z�=s�$�H%b=��(<���;0�>/�8=��g���պ�]�=w2�C{�=mQ��ޠ�<�¼�*�={�սuT����>�<�zY=���>�B�����ν�׽{2�"����;�fȽ�>D�-</>�q>��V��$�p�=�#�Oڰ<��4>��<�������1G�;���C{�=��<c^S���=O}ּ�=E6w=[����\L�	����q5=r�Q<^�=c*G=��='�b��Il��x{=G�;.���f�����P=U�ѽOW=K4�>�]<Z` ���9��T��dO��w��zm&=���]8O=�e���T>�"=��C�4� :�6@�A�a���=�Na>��:��n=.?s�֣�=��ݽLq>"�V�u`�=�ܼtӦ=A]��pS*��S;;ND���c=y��p�T>Dvv<��+>�O�M
�=�l���_={炾�6=�=�4�a;?Mb>�W�~xK��X�<Lk�<X�q<�*�=��ƽ�N�<c�@>�5#���8�'��<(��=F���A����>;�L��
�<VX�;*	u=�V<%ü?��;�)����P�mZ�<
F��Ҷ="k�����=�!��<ă�6=މ�='e=<G�>P����=D��Yj>������<�h��3[��<jV�~������<\R>s�1=mEo=���v�=���:�{�:2f޽c�a=ۘC��ݵ=߶=��<��=�[ѽG/>7@x>+px�L�Z<y�мJ>Zռy�l>%�ѽ�`�Hh����=-�����=!	>S�k<ܲ�<ȫ�>}��=�}����S�)J>��;;�=L�� ?�<1
��0ې<#�|��a����m_缳=O�g>2�P�<�Y�<nn�=��;μ�=�.w���=�=_��;,�=#fh������<���<]��9�c潤�<�|a<��=�Ӻ<(G>i�u����:y�=\&c����o/R>���=�6ս��>oEM��IG��Oм����&�;)�6BM��N���mdB���?>R>�a湰�3��`����/>}=V=�=�fܼ�b{=�jh���l{�<T����P=DD�Du>>	�#=Q���[<��,=z��L =f>�j>6�.�F��ݾ�w���I����=,@�5�L>X�<�Y��������b���#w<ɯ=�5Q��j�=�6�;E��=��=mu#���e={��c�7�O�8t�>l�<o'�=K�=C����O��ec=O��C��gMm<����=��)�=��*��s&=�T�=��>t�b<�S���kn>hc��3�=`%t���b���<���<�< �f��<��o=�p�=���=�F$=�kӺ�l�G�<UyC=�m�}u>������<8���<�潐�6=�&�=i�Z���j=�>�t��q��<�5���>L\��.=�=0��<�!4=�_��P=GA��P�= q���;�!=8X��N�=[[�<}�>�g3>C��<���<�:�=J��Kɼ9���\�>�6ͼcm
>}��&:6=��=B�>U�=�DQ;�n>L-�;�->=ֱ�<�=�<����#X=�?�:�=�ߓ�t?=~�佃�$�����o��;�E=�ɑ=�O�g����= )��(���t��$�������==���<@i����'����9w��<��Z�џ!>W+ =N��<1��̾���>�"R=6}i���3�KW����� <>I�����>Ӻx���n�O<�;�X�=� �h�l�=�B=�^Y��0&��
�=ɷ�<�ܜ<c���S��r�1=� x<��a�&�(��	X=��=^h&=[�Ѧ|;6�ջ�y��E2�<9�O=����4\�1s��.��:᧿<�V���~V��V��}$>`H����3�������9�"��q1>���Q���.7=�T��Ex�=�s6��B;�����7�SR�:R]��(&��g+</��`��)7�IF�<hI==@}�m����D=oYM=�4���P�I�q��e>�Pp; �R�N��<Y~=/v]<=Z�<π�=�u��H ��`B=�D��'%��oc<�b(=�ٻ��=|ώ:���= x�=��>u+��"�=ѹл��'=5��<������=&l�Ӗ����=��V���=��S�=�C�(:<<.m�<�|�=PP}�I>*_���ev���<bK�]7=ם)<��r=�F=b:����]����=��d�P����<�.�=�<`딼lJ�����=h�)&<9��y誼�;=l>(� }�<M�S=���=��%��"�=9`=�[L���<P�K�d(�=w��=�N�<�[>�P��� =�zѻ�dN<�����@��<����=�.�=�&��7˼^+<ʴ%=�OP�$��L�=�?�/̱�.�^:�f=�E�kKQ=�y
���>��=��=m���Q
��^>>%�F=!�=�N���G>b_�M��=�eA=64�=@����<q%������b=�X��%6?=��2=
�����ܼV2�<)�
>�J���=W}˽�����1�<s(�=q�6>��Ͻ�*�=�6���!�Bj>���;��=�K��a<���ڽ�Ө��O"<[ν�б�A����^=mz��(,�=��+�=��=�u(��R%�|ɿ�4�.=+�2=�8��렢=��2;3g>���=dN�=�_>=�%�=��=i�=���=��=�y�Բ=�V��e��=���=a��M�=pP�=%���>0l�=]�=����Z��A�=�ue�9�/�=��5=�[������t�Hɒ�h;�<0�:��+�=���=��=��>�+���P=���<P�<ڀ��d$�:t>~V�<|U=��;Z>9c�M��=*�k�<R�[�)�h���k�<6�<J�c��N ���<�;�<�Z�Z~f={�=�UX=7pd=��;V��=��-�>�?a_<I>�=���<k�f��&��μ��n=�aR<�A4�9�9=8X=� �[��<��P���3��Ck��K�;8mý6�� ʨ<_AC��;��A�ƽ�g��b���@��=)o-���a>%[½B�)>)�=�[�=���p*��K>�m�<�N>v�W��%��������ȼq���������=L���S�=���&����fͽ��(�0�Ǽ���q'�=�=������o���:Ja(��ә<��=Y!d��Ȱ=Hb)>�R�v�E���=<ؔ�=�[<��S=l�<��"O�=��<�ϽP	��m��=�d7�3�<���%)O��p�=�ք=���=�-=�_�<��5�n<������=�\G<!��<=i0>�~ ����=�¼=���=h�;,lj=�I>`�"=��4>�z�����f���U>�d��hyZ���o�{a=>�S�=�y=�MY�Ϸ%��%H>���5�>�}�=�\�=��u<mM��'�=^���ޘ�=k�> d���Kͽ>վ�A=?ɣ<T�;?ԩ���~=U�=����S���Y�$>	#>�Qi>�i�>�BP��н�@��Z����r=x���d=Sh��C]d��w��ʭѽ&�=���=�$�=��>x�c<r}�=��=�]4<�h-��G`=I�C�>z��|�!>s�5<��=A�e<	�	�Ͽ��z�=�>��9=y񀾀8 ��|�Z+=��=*��n�=gc4�uE�<�VT=�y���x�=ȟ�I���$UB=l�����=�m�=�A�=F�<�A��J>���<��>� n��/�=$@��M½[�->K%d=��>���=2[�c���/ս��:�t�h�N��q��+����ν��t=��p�#pȽ��!>dFv=��>�3->.�i��t ���A=����}�����᣻Tf�=�u�<�	=H��EPi=��=�.��ě�<0����M���
���㾽�Vpź�B=�T�Ԃ��]ր=t�=>��aۦ<0�=R�F>�=�A��bv���ҽ}&�?��=�,��cm��x�#�'<��=Zah=��&����m=D>;;W�<<H���$ϼj]�1ڒ=4�;�K��;�B���<�����-���&=��b��i=�j=���>�=�u9�'�M<�o۽�ԽX���~C>!E=�a�<�0<�|?=?ټwL���K>�
>��>�X>�c�d�l�5_�� ����!��� !���K�ۼ-�������}=�\ֽ�g>�QP=���g(>N=���=(ݎ�0�5=4�>�`�$R>��z�p�=�@�=���=+��0޹�%}=+��<E�,=�һ�=��t=����v|v�T��<"%����9�~<�ZM�c[ѽ/$>Ơ;Y{L>�^1>&5�+��Nn��3��14�=8�n���+�jj��;9<�K�X��<�>P��l�*:`5����	!�=��b<��=j[�=�v�6}�>�5��m�>(:g���=Z���Yq�=��=���j>V�P��X>^�=zp=�鱽�1�<U<=�;<�Ka=	(�����=/8��Fϼ�
>�l�=[_�=�*>��@��/ý��/�ށ}�w��!�\� �����y�=�
+=6��<B�K�|�>��=Ƅ;>�x">��=�Q�=�q2� +`���=Tt��֖�=.��=BqѼEd!=���u{=�,�;�^A=��=<g K�_�L��n�s�;d�=�jW=�4ŽK��)�,</*=��=M���{%���=$ř=\V�=��>�Z=[�=��`��u��:�2d�o���0(}��ZF=]��=����v۟=|R=,F>���=�d�=̓ ������꼦t�=���=~��<�n���_=���Ņ�o"6>In=^0�>�-�<�$�<E�����E���O�r�a"�<�_�vP��=\~�=׉���/��(Y��[=��c>d�>Z�1��;�\��<G�<�0&��)�9m~���#=/*���=	�O��R\�ڒ�H
�<PX=�;�=�AM;��T��	޻j=�5�<��c=[kg=E��Π���J}*=��1�'��>��>ǔI��0��[�"�Փ~=�`<WC�:f�=aå����Q�:{��<��:���F�eh[>�l��1Q�=�t=؋=y�;0VW�K�c>���<BqV>�܌���<�l��PX�=;-Ľ5�ͼ
tq����<��b>x��=,Ŷ=Ø�[7!;=�
>}� ���->$���d>�8�Ө����̽	N��<o�J�&@%>�,=u��='�=�{�=�W�=����4P>28
6StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_5/conv1dConv2DSStatefulPartitionedCall/mnist/fc_5/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
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
:����������2S
QStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_5/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_5/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_5/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpConst*
_output_shapes
:*
dtype0*U
valueLBJ*@H�ͻ�=�F����<�<�l�<S��=�m=���=&(=�C�;��#�=��=,sܻ#��=2;
9StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_5/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_5/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������2,
*StatefulPartitionedCall/mnist/fc_5/BiasAdd�
'StatefulPartitionedCall/mnist/fc_5/ReluRelu3StatefulPartitionedCall/mnist/fc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������2)
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
:����������26
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
:����������2Q
OStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer�
6StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1Const*&
_output_shapes
:*
dtype0*�
value�B�*�M��<\GR=�W�;ŵ�w��=`����|Ի���m������i�x��x�K|�:+����'=������3=S
<'p��c�=��<GA=T�ļ� (��"�<Dm˽���V?>�m=q�>�KԾi�=ѳ4���W�%�����@z�ѽ�<�<ɗ��Ž�<w~�䬼��<Omr���<�aJ��'��E�=�V�*ٯ��<�z��r��=�m����qD�=�˽��F�W�l>5=c�=����V�����=c�m�|#̽@W��-	�=;'�Xk}=����E��{k =���=f�V>��&=�սk����=��2=a�E=�tX�S�	=T�<�����!<���+�^=��ʽcǣ=��(>w)=�����n��ں<۩���1<�]�;�hZ�����R�>��=��B��X>��X>�	���󅾰�=���<�Ά�L��<D������ܻ�=��Y=�����J���p�����F�l�	=�=C�%���k��������=4�ǽB��=*�:=��3�:��<
0�<�ᢼ�W��7�����|�^ĺ�4þg��������>�8.=r�=��>M[�=c6�G�'<,���� �=�]�<a��<�)�<��I��������P�������=`�^��0����=�se�쁽3�I�\(��~1�H���:J=�)d>~� ��+>�y��g�;�)<)k�����.�o�;�<(�<�	�%a�<����S`��[�=Qd�=� >U=̊����R�Ȃ=O�-<@Ly���?<P�н舄<tuP��_ѽmZ$��ս�?���c�=-��;�ذ=����<z卽��W�=:W<�����/��4L=�;���7=OM�����=YI�����>���<2f�=+'��O����i�e�=`�<�{���?�&!�=A�?=א0�7��<.޷�)'�=.%���̼�ֽ�q�;nH�A
(���=��*=�ŝ�0�:=����@I���S=WP��^�>���!ڪ��	
�����h>J�s��ɽ�Q��;�l=�:ؼ�a�Tv�p~<�W�l��ԥ<�P��ݜ�6��<Xm��=�?���n��b\�XY�= so�9B=�<�6�=}�/;�֘�'���#��8=f>���u<[�x�q���=�-ٽ�3���`=�t=���Tf�=K^B��@���#B�п���=Z/��	�R<o�<܊<�d�<�BI=�zλUcN��
�<�VĽ�r�=a�5��ӝ���ý�v�X��=���<�d�=X9��Q���_R=W�=A'=J�>d�x������ه=�����{�TQ�>�.>�J�m[�$,��s"��X��&�2��)=�h��ذ<���=r
�<D!���<H�N<`�9�� �^���'4�=0����
��=ڞ�òS<��>����YK�܍���=��<����_I�>Q>D؉���̾Z?�;"K}=��ݽ>��:�&�<��;�^�=���=#��=���=���@Y���->��2�����=J�G��� �=�2ͼ�!��$e�B�?��"�<|.���;��Ze��O$=9�ѽ~���>+ʼ�Gr�&�b=��=1����Y`���;'�p�oB��o|��鷵=L�q���=�zW=ӏ��L)��E�>���<���<ĪR=��J<�����(�eU�<�n�<�v����`�Żj�N�	��;�ۗ<Aؾ<�F��CO��N�l�.�=���<au���Ӏ�nT�=h�=pI;�q�<
qf����I鼔ٻeFR��f=�̫��5���O����½��.��]=a�y�:��<O �M=殶=���/}�%��������h�=mbz<-���p�=�%��`\_��@�<R|'��|�:��x�8;��G��8�W�����<wc�=Fب��0ý^�=��̽�O�<�;i�3��[+=#���	�7ۼO�</�W=��$����=<�s=���R>���d���e������v�;�Rv��{�$ُ<}��9p<����N+۽M�ڽB���d�<=��<y'�=�ah�2���Lj�'c�QI����M�n��<������<7�~=b�=Lo97�O=��^�5r��<*y�=��$<x��=��>8� <��=P��;z�Ͻ)o�=7!�=CA�:h����=Y�=�d�ƻ�=xS��4mE:� �<5.���Kz=[Gú>���⼾'
���ǽ�Ȼ��l=��y�΍=6Rz��)�=Y�Լ���=�G �X��<���<�A�=�=GD�-�=Oÿ=���<b�;D"�ZY����������1=�=ip>�'�e�=|G=�D�2��p��%Հ�J=����Ҽ�]B�M��b�>:zE�5��<�et=ou=�s>D<>>Ml/=fP=(�=�e~�X�=���eA��%��=�x(��-=�7���'�
*�=�_�=�K��MF�V��� ��KV��v�<{n'=�Co��*>C������Q�>pj�����=)�n=���=�+�=�>9�=o��<mଽ֠�<2�=)�½����6�>�灾�W.>�B�=F�v>_1#�Q���H<��q�� ��潠
��򅽠�P������k>;�y=�9�=oB�<u� >MJ>���<Q	ﻝ���(�=Z�ڽJ�o���i�VhB<4#'��� >�M>��
����<+=��>ר[����=�G!=m��=	�I���p=��=
��=X^�B�o>�7�=!u���=�F�M��<��^���h=����d0=�q˼�WW��^�sa���R�r�=��n=Y!�|�Ƚ��7<�jG�^%���,ڼ휆=���=�=�"�=lg�=;�Y���U=m`)���>�C=�=$�̼Y?8�ed�=f[���x=n� ��J��R��<Nbu�p^�!����k�����;Ts_� ]l=�	=�rR��}��?	˽�.�G��]����(�	սUl>��G=�ڿ;�3�<�)����k;��tO�<͇p���=C��⋼�@��9~Z>$�̱½K�\<�͓��Ge������c�<C��<�6�B�=w��>ȅ�=V�=<���$�=28
6StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1�
)StatefulPartitionedCall/mnist/fc_6/conv1dConv2DSStatefulPartitionedCall/mnist/fc_6/conv1d-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0?StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
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
:����������2S
QStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
1StatefulPartitionedCall/mnist/fc_6/conv1d/SqueezeSqueezeUStatefulPartitionedCall/mnist/fc_6/conv1d-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������23
1StatefulPartitionedCall/mnist/fc_6/conv1d/Squeeze�
9StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpConst*
_output_shapes
:*
dtype0*U
valueLBJ*@8l�<o�����{<P�����¹���R�<X<D�gʻ��.=��P=x�=�/�0��<�j�=a�J�2;
9StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOp�
*StatefulPartitionedCall/mnist/fc_6/BiasAddBiasAdd:StatefulPartitionedCall/mnist/fc_6/conv1d/Squeeze:output:0BStatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOp:output:0*
T0*,
_output_shapes
:����������2,
*StatefulPartitionedCall/mnist/fc_6/BiasAdd�
'StatefulPartitionedCall/mnist/fc_6/ReluRelu3StatefulPartitionedCall/mnist/fc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������2)
'StatefulPartitionedCall/mnist/fc_6/Relu�
+StatefulPartitionedCall/mnist/fc_7/IdentityIdentity5StatefulPartitionedCall/mnist/fc_6/Relu:activations:0*
T0*,
_output_shapes
:����������2-
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
:����������2/
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
:����������2R
PStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_8/MaxPoolMaxPoolTStatefulPartitionedCall/mnist/fc_8/MaxPool-0-TransposeNHWCToNCHW-LayoutOptimizer:y:0*0
_output_shapes
:����������*
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
:����������2T
RStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer�
*StatefulPartitionedCall/mnist/fc_8/SqueezeSqueezeVStatefulPartitionedCall/mnist/fc_8/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:y:0*
T0*,
_output_shapes
:����������*
squeeze_dims
2,
*StatefulPartitionedCall/mnist/fc_8/Squeeze�
'StatefulPartitionedCall/mnist/fc9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2)
'StatefulPartitionedCall/mnist/fc9/Const�
)StatefulPartitionedCall/mnist/fc9/ReshapeReshape3StatefulPartitionedCall/mnist/fc_8/Squeeze:output:00StatefulPartitionedCall/mnist/fc9/Const:output:0*
T0*(
_output_shapes
:����������2+
)StatefulPartitionedCall/mnist/fc9/Reshape��
:StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOpConst*
_output_shapes
:	�
*
dtype0*��
value��B��	�
*����x��xD=�v*��e��IH���;4=�͔����ޥ;��3=�Ӽ�"�<]r���o=	<��������	=G�;?Oa�������9N�z@<!Q=��F<s�����<V5޼速<Vv=�EJ<�K={��vP��{ú9����sd='!�<�W�g#=�(r�B�$=�Z#��M<v
�G�к��1�z�=������<G��ʖ<���<���<&�+��g1<�*����2=�̅<3�A=<L	��H�<D�X���<h�c�=n<�(=9Q>��+(=G�޼k�ۼ�u+�~��\Ҽp�*�5߾�%�-��?��Ӻ<Ȏ#�4zK���<z��<.��<`�;��'���%�Pqv;&$��rS6���<���0<�J�<$�!�|�ڼ^F*�_X����<��<Zq=6k&���`�n�=G�=1���Yp<ڹC���B�f��S\;�v~=�Ā=G��Yq=�I*�����?̾�,�!��<~�8=@bG=]�^���)�763�@��7B3��,�<ب�A|��N$<"~�<�I<&	�T@�;��f<)z<�ݼ��<����0b��s�=�i =�}{����<�F�R�l;N�F�B0����'��F<e��<�I�<A@�<��<K��<�4���)��H#�zdW���<�ݼ6|��`���M&]<}�(L�����<��4<���x�7��+�)0 =gcm<�O
=1�;�h0�x�ٻ����:t�n�fN=���;�q�<˛��}	�<�A<���;�=��ȼD�
;���<^N�<�fJ<	�Z<h��cɻX�Y<�Y��+<�ܔJ;I��e'�q���J?=�7)=���[�
��<�Ҽ��=K��<�pƼ������v�=1�]���/�赃9�6+���6=����췼kar�-�K�<	����мo��<x�������_���L����禼VK<�Nּ�q��!�n-<&��h��>C���
���	=��<2ż.�,����S��;r�G=(�+��3	����<9n=�F�<���;��z��ӓ<T0�x�!=TX��y�B�˂ƸLLU<�x���;�SI��y�}�*=��:=z�=�96�S��\= <)p<�h�G�^��6(*=��T�$ڀ<\Ѡ<�y�:I��;ֲ3���<3^�<��B�d�;���;t:=E3�,�L<�K�;��r�}<���[�����drz=:T��/Ƽ3�=D.�TK�;�2���=���;��H</��<�뻼y>�<�u:���=��=/ �tP�:�_8�+o�=�
��8�<��<s$� �Y��W�<���Sq/<).�<�Ȕ<'��;H�<��}�ȼ�-=���� ��p��;��T�30<�
L6=G�*=p�<���C�伉�h<��6�޼�$�5);	Hw�+�żpғ<Sp=�<c�=ߡ;�`<ui�<P��<�����	�#P��C������<�I�<�E�	M�<�ϕ;S"�<4V� E�<Hʔ<��<!��ќl<�
��<�dG�>� ��R&�dJ>=K}8�N�@�:n�<�~�(�U;�����k)��8'�&��<T��<ϕ���F�v]���Vм��#��@~���輸u����H���;J��� �7�b��<�	=PT̻��/�QZ/�Ῐ=�	��s5�(M=�@J;�ŀ:9��w7˺��<ɺ�<.�<{�8��	=��m<�=���;��i������19���?=�F���<O��<�9�:f�<��軑7Q��Vh�I:��3S��՘�����M����%7���<z,j=�a�<BHɼ�C��!�<b8;E.=-��`�<�2=r,O<<�����.�:���Q<mN��z�=g�#=��J���=�a��;��I˔�rH=�,���$����<$@�����8U<��=�;�0$<O�˼��Լ^P< '����I;��!��c����;�y��o�;��ּ �X��l�<I*L�u憻3�U=T`=\�=ߣ����<�Z4<��l�8 �l}�<"C=+����<=��:�i�;���;�'i<�]�<�dv���< <D�ϼ#k�<&Tg="#<�R�;��/�&c�K,����L�[<�[�<6o�<ğ�;9�o:&����&	��S�<9���_h<�"����<���C)��d==���<��˼�����9�ɉ�P,=����ӫ���-ʩ�� �D\��U�<9t=��	=<-�<#n,<������ؼ2�
<u["=�������<}B/=}��0�t���j��|�1�ʼ*�<���-x�v�<�D�<�^��°����ռU�<#<׼���S){�v3����q=W�S`���b=!y�<����	��;���}����Y=���z�!=z��=��:s=B�=I�<92��
���U[�-��:��=��:
�s��z(�<쀙�FCļ��<W뢼&�)���̼� 2=g�U<H�3=(�t<L�6��a=�L����nv�����-j��8T�;�,<F� ����<Nro��
<�~&���=F�<�Oݼ��ּ*�<]B����.�ZJ<��ȼ��A�<����=�5=;\�;Y�����*=����y�ʻ�f�\������<�F�<^py<2�<�M=~c��e�&�o���
�<�N��h�i<:��<�V��E���N�=��<hV�A��<�<c��O��.O=�T�&6��:>���<�c:m;Y����;l��U��X0D�~aw��KV�9���C<��g�:3;W� �I=ļ���!\��r�<�*���[J=N��LŻG�>��z���/���ؼfq��VF�a�=��7��(���4�J��<�x<_�;�'=F(���Ns<������,�ە<�];��";� =�����*�����x˺!7-=N�<�Kɼ��7<g��;���e��<�d�<�������E��[�<�""�>j�<�:K�=�=���<Ӫo��䀽��;�9<�|��J��<�[y<��e=�����H[���j"ݻ�0t�9a9�P��<:,޼
%1����<������I�\`�<�r�<�o={z����0<ˆ���g��.�:��1�R�!���#�F�q<w+�<��мB ���b���=�4=���tXt=��Y���=7������:� �7�C=�|�<�&r�C�=���j���^�<�g=<|���R�A��<��W���s����΁��x�isS�!=�	o=��*���u]�����Y��:��%=�z����¼�;P����<���rY/;0K=F�(>��	�<�Ӿ;'62�:��<6���C�H��{��<�v<S��<�\ҼP@���!=x��l6�<�0��P���B���<T3M=-=�,���.$����B��<�%B;���<�&4=,~�<-���a�<��(="f�<#[����[�i�+<���:��� �<��,<�@������=W!�Jv�<c�*��<������<��X�~��<�V*�FZ���"=�&#�4(s<?�ͼP�v��۳<�3%=p_���� /�P".���r��)�=<�=W��qa=荤�Nc�#��;8��<yJ�<�/�<JB,=VP_��--<k�1<�r�=A6��И��@�g���,�\�1<�?�����<��!�'	Q��R=�<1��<KWp��n<��C���9����O���3bL�S�<�A=ϼ�z���=���;�\:� cP=[u��@�Q^�<oN�<@�%<K�Q;4�(���K�@�E���k�O�(=�#㻪��<\s,��߼+4�j8��'<�Y�=|z=��=�Q�r	���K=�sV<�~�; �*=:�u��h�=z蓽�t�X�(�V�=��<�<Y��K�E=4 ��*g��3��O,=����N����\/=���A�ǻȧt:�v;�䄼i�<�޼�<1`Z=ƛ��<5~#;��<Nn/�\�t�f�b<ꫝ��:��X:^K=�$���rz�<��-��Q_=�^N�"W�;�����,;�U�<�:�<u�,�O�;S���[��� =V�N���貢����=�?$=V����ܒ<��)��r<Da|�1��<傹�k�����|=���-��޺<����.g�<���<�v<߫��x�+�G�˻-�C:zp���K�<�P=�?꼼	�����4�<�&'���,=��.�d�V�])���輙c˻M`�<�(:�I�m���;���:��8=�;�<��=�<!��-=��	�=��=Tܒ=DCd�u�f��d��T��<�A=#'=A�;/�����t�Ij�<.���T<���EQټ,�!��Z���?L=��g<-��<;�_�<�u��u��¼g�<������<0p<�?�<h,���37<��"�lμ���<}�H�P���z�=� z�W����O=����n2��;�n���8H<p��<�!����;-�(���n<��'=�7�����e
=�q���{<C\���������z�Ey�=�������<�jx��S<�;�<����[�d]�����ج7>�:���f׼?M<KT��YV�x�<o�;�[6@���X��C�;~�;}>��#���s��:�(��,,)<�"�;6��@��<.C=;�<MU><+�H<nP�0ċ�+E���=�V��͏�}E�=���;c�	=��=;cD<2O�q����|�ʠ����<���=!P�{iպ(P��Kk���#S��2m��E��̿�ŕ���>����(ם�b5�R��<\\�����<T���_�<�����{�`\'���<y��;*$�4沼>��l䣻�2�<F���:<[�������	=埻��<*��t���a�h�+=��=%
�<6����V?�-��4/�<�{'���_<�����T=Eď=�?A��X�*��s�y==R�<M�C=9�����<L�c�;e)=� �N��p�ڲ�]nܼ?x��J��"����=e=�<,��<����.ǽ<���<�UN=ћ�<��<�hǼ�����h��:�<p�-�e3���x��Z�ʻ�;=�r����=.�;��t�����?���лrt;���8B�ݻ#�J��s��6�=��������d<��*��Q�=��s���!;���b�� =≺�w�<�[<��6=�y�<ulk�����g���K��T+�<�*g����
�ȽR&�<��\���=�"=?VA����;�����d�����n�N,=>�<Bb߼�������*�;^�=w�H�!fU�jRk�f�=n�>�'� �[��ݽ�T=���Q���3�<ƀa���=��=n�%�Z��3Խ�@Z�F�<C
�<n�Q�k~:��
!=f4�<������̈́���ҽ�B=�A������T��ܶ4<B[�:dC=�K�`<�9><�|?�`<:B�|7����;���I��:��:��<�4�<��%����^���N��ó=�@�=��y<���O�̽�;%`�;w)����%�r�U'��d="��W.��n�; �h=��=�������#[���*<wF�<�>��:���~|�ѳ=�F�;A�=y﮼�2a;���;�-A=���Z��ｩ�S1B<B7�=�빼q�ɺp/[��F<�ݼ{6��N�?=<��;K^(=�`"����<�*T;u3�<���z���,4/=���;x��d<�GX���缄?<L�</��<[��<�e�����	S����D���<γ�[9t<J�L���=�;I��=�ȷ���8���AJ=[l��'l<3���F=��׼3�=r�ŽJ����[½�=x[9��?�Ͻ4�;�,��;��üe�<�:�< �;?Bk���&����g��<T��2�p=���K�f�.����<R#X��������<i=Ą<�XF=tg���ͽ�
<Q�"��&�Q<��P�^*W<��Y=�Y=��.�����y�5�%=I�����a���Xa�=è<�6=��ܽ�
���e��E�;E�@���O������D���(U<�Q��JM�_�]���[;
=�<-����2=Q8����<���<�i�<(u!�	�z����"=��S��%�;>t��gR=�&��$+=�=����������<��� 48�����:S=W�=�;_=��ڼk�R�D��<!_弳I߼2h���酽�M��Q	o�m罰�U=�8=�����=Y��A�k��"E�p��=a�b=��&��k�l��w'����#=7���'l��U�Ľ="aмf��;��<�g*;�ļB�<}G�<�vS<��<nv��˶����;)1<h�9�V=1}t���R���.=`9�;{�<<~�R=�K�<��I��ڪ��hʼ�iI:N�g���m����.�<�$=5�=:�����ؽ��Ǽ� =�8�_�#���9�<�U���W����O���ͽL2�`m>�-=!N�<��;��5f���w�)=�qp<<1�N��<"��<=��<�@B���+n0=�3�����<a�������@ 8�o�=F��;�O��}�L�s��������;2
�����6	8=�~�=c�g<N�H�כ��?R��n�S@(��kN��ν̉�:=㧎<�Iw<ꊄ�]�d��G����fM���	C����=0Eؼb�!��U�M������~\��pE=���;�I�_��d��#�;]�:<�<A��;���6{�T8�<�i�<���<�f�����<j�ټ��ѻ���<z��k1����&0= Y>q�ŻB�=���5�nI;m�i;�wj��X:x�ѽ��<��=�߼�����R��g�=vUʼ����#׼�ǽ9��<�;輪JF=���wݽ�8A=�u-<��o=��ܼi����	\���!=>X�<�b�<D�ت��HټKyv=�ޓ�s+μ;*Ƽ�Ef�m��.�<KL<S��<��l��1���~ü1�'<�XB=� c<�j����,����^=� ���=&���|��f�<��=��;��-�`(��C��<䏃<�%�����:�/���X4�qKk=�<s�=�9��v��@�5���˻8d�<(=��"x`����=����7�;�c,�Kp���9������>	�O�;ZEj=nμ�*�<�ۣ<X������<�)=mӽR	/�Ar�
`���xx=�73�[%V<✳�.�D�Sq=Ξ�� <@����!����<K��gR��;��0��<��G�r�h�9U�;���n���g��=d�9�o:��5<k:B�Y6
�j���8=k�g��m�Bw:=N�-=�m%�&Iһ��<�M��'�<��
�gze<hD[�É=t1`��:}� ��g��|�<Fm=0�M;�	�=���;q��
�ͼ/
=q��<��;=�[=[�<��<d�`=&���#i;��ֻE��x`<2�;S�����Ry漎��=���=~)��k��:��``�=*�ɽ�D�=d��</M	�+^��_�A>�a0�t�D�g׺�m)D;�1��������8����JA����=��<*�;<����<������='
���@<��<�ND��֯;4���m� =&b��^��u��=4)k<}��_�=��y��=��U�lY=�j�&���af���_<���e[����=�g�;ȉ��+,�F�"�Xp=L+�V�μ-Π�䛳<��;'8�<Ɠ5�w4|<H�<ik<����N<��v<k�źUЇ=�j⼒��<�`�;oc���U5�q�<��L�zì�;"�=����G�<�N0���C��Tk�����=�����Ϊ<<5$���='L#��K<�Ӽɸ����q<��<9u�Y�ż�H�pOB<$����(2�T����N<����f6=vh=[�-��B�<ܝk<�a4���_�&=����&=�#�Aj�j� �&�߼㌵��G�a	��+�<���<Dx��\�<��<e���ɭ=r���hF�j�`<-�,=_�<Z�����{�a�2=6^�<�K��D-<�=+�p.�;�˻iB�=��*<��<��t���;d7�=�<E=��x�=��s��f=�5S����� M�=eWv<,~��5<Y&��r"�,����mk<��	=/?=�㼽��j<���=��
;OuC��꣼3�<E&��^w<�w���W:���I<�=��T���U<��<2��]�;�
=�
=u���\ǧ:��<>�<ƙ%������;;yg;���<ck޼�$�A��;��+<"�#�:��G���)3���=R�=F#��f	=���<ex��ٗ����<��)<(�A;_�>=��e���T<	5d�P�=�2n=&�g<m�(&μ��p��`	��4�=`}����;����c�<jB)�(%)��<c��	�;�&���;;[9
�~�<��,=�Y<VTO�[=��缐i3�]��<�'�<�+�m����z�%� =�w@�����t��<�z:��Ĺ}��4˻F}��-q<�[�<m"<��&e��}9�C(�;U��<���:���H�|=PP����(�ӎ���a+=�6i�@r���Y��!�'�Da��О=D����t��BR;��-=��)=�<=�G�<=�=iݼb��:�s%=㱶����;���VH�<�Ɩ�S�<��_<��X�[,��#� G� o<ne<u7�#�`=�9ϼ'Z+�l�?�yy=
_�<Ϡ�y$�³��Tf�=ʂü�0�gp<�р=wԍ=Q�N��X�-B��Im<���q4�<��</�<��Pʼ扺����<(���X��Y�<��<��=���:n���<���3����k�2�<��=�:,ϼ������H=��T��.�gC`�M�S����Ua����e=��X=b��U�n=�t�;�M�<����m�F�����Dj=h��,2�<�#G�S�T����<"5=�*�/Q�a�~�ʎ�I;�<ӊ�M[��5��F=V�&�Fm��/]�����<r�+=N�������2����=���o��н��������K�<��=��<:�=�̼���{���YY�C��:7ck�i�<'� �pP��a�˽��+=a���Q�m�
�#��d��,��gW�<'!L��
�E�?D=P��;�́����[��a,�Yx�:�n�;?tʽ3�н���<�:�ͽ���;�bϽfԽBK�=��^;WO���ὺ<�=�Ȼ�q3_�3�r4;n�i�'�*���l�;|�(=��K<��<+���J=����J��BƼ��='!'=���i�l4�<!�C��=����.=��)=8��;��ս�x��s<y�h������ʫ�:6��@`��l�=�t�T��x<�A׼�g~��V¼<���<�����;�p��K�8�&�����=��I= fH�P�B<8|����)��m<�Ǩ�`��K�k��=%����s�(�:(=KE2����<1Y�:����k�Hּh�6��Ҳ�:����;�S��+�;w�m=�����o<��8��H)<5K�<�+<;]x�}��v�= ��<T���u���< �`�]o�6#�����쇛<�_�v2=�~��3A ����<$������3�ҽW6��RK=r�=};������_���Վ=h-��K�L����ԛ��=<�.,���'<��P4*���<� �<x�j�\�<^��h�żo`<M���dm������T=�k"�a��	���̲ؽ��f=���=�y�ZR���Mq��$<	��_	��䝼�3��8<ػil���w�^uO��J�ס��e�۽��M�Xq9��>����㯭<�s���,�)�!<Ԣ��~þ'e��pZc<��=Q3d;ᄿ<�����<ܑ�x;�91���^�#�|<�q"�è��-���ms
<&���5=
�[�QuG<�(�:����%&>
�K�i�߼~H�����9�&]=|ħ�,���˼���ж=�+=U�=�-&��*��;N\����K7B�dĮ<����FO>��P<��8&���	�Vy�<�X�����<��i��O6=5��=O�3�#��	���6�&<�+#�W���D����<0��;�qd����<ewB��9k�zм.韼,���@O��@s�Eu<^�E�jkE�����ڴ =�jU�o��g_=埼�����Ș;��<����K�����;P���ާ޽1�,�_O-��=z�=}��g�Z��@����y;�����ܽ���J��n=�6U��
��cl��g����=�����m����w���7
��{��	
��M��¦�<�%=��4��-��=O<:���w�=��C=D2Z�����i�� ;�Q����������Ƚ��=�O�"�z�5��B�N=�Ab<\����k�����N1ʽ+@<Z�<Ϝ��&d׽t|�l�-=� ���Y��|����+��F=ɾ�ܱ	���	��<�;>�;s(C���zxӽ	�������w �����M��<��=�<��	<���:��y;�d<q�6�ǌ��b2�<��Y�ک<��<�I<����0��<�b��=�,��$4#��6��	M;�f2=�W˼࢔��#�I��]�=]9�<F4��gً��W5�p�=�[S�"綽��������̼����}��=]��\A?��\=+v/�
�����w������&<�<��_���c�gH�ς=�`��)��%H�&�.��1$=B1%=�=�k���!@��/>���<�W����;�{�%*��$y�=b~�<��2�s�S=�e���@�@2=:�;�i ��+�=	^��u����)��K<b <��ȼ.W��yg��%���c=�r�����I�ֽhﰼ��=���>����{�V̽\`=$�H��g.��6C�?f>����=j��"�<�Z�ց��Լt:Y+�;�1�Z�K<����ۼѺ��$Q���ӽ=g
=T��ګ_�T ��=�<2��=�)��h��v���-5L��#=�]5��}��K���?�=g�`<<�B�����
8�ή��6=�:�F	�����{-=q�9��>��Խ㲅��4��)y=)n<�� �����-��#��=:HZ�������r�� =^I�WV������&�l�<�5���u;��A�<�,�p�<����F.�<M�����x�8�=7x��� ս�� =�̴����o31�w�M�,5>p(���j������;ȽSVK=�`�����V ��F��e��=Ln
��;5G��a ����=Xw�Un̽G"��#N=���q��m]c���s����A{a<�`�9��$�����?�ցv=��7�妩�g���φp<�x<�Z)=�U���51��N=~����μ����Ys;�Bh@�r�J�&�J=|q=4)o���*=J�����<� =O���.��׏<���������񽻘�<n�=Ԏ���6�`���l�?�EF~��üH��!�y�	��O�=.�l��h���-V��d����ּ*5=娱��Δ��b���ګ=�5��ꆀ���9��4����o�
b=Q<�q@�'�뻧��Y��;Q�<,I��hK�G8�J\�;��j��o�@�\����<�K�:�#��w����(Q<�a<~_<�Q�;n<~�=L�ι\�Ǽ[�d<2�=���*�8�;�'�Ƌ�$X�����7H�<B�n���c�gs��4¼3���+���+��AW�RN���Ӝ=�Jx�l�����ݻ���=� <��ƛ�bu��O���=<�K\;!d�T
�E9�<���@��<W�<�桼���;a&<0x<D�Ƽ�d�Zz�]�n=�@=����`�/��z^<Sޤ�Sx�<��M�9ғ�������5�U�<,4��DX���l�;�J=&�ۻ�X���c�p齒z1�q���i~R��Q2��,�<���=~��v=^���Vo�Dj�;��.�>���
� :ܼ���<� ���ż_�:�&}���<*<*�9=����3����`=��i����4��<i�|<�P��X~�>_�=y�i����;�J==���������,����<q�����;����d�������[�=G�<�e�;Q�`���>-=�n���ȇ�9 �w�9���3��i�;�8�H��;5o�����<˽��+e=�6O�:_<XN�=�`�9Vj.�Y!%�k��>z<AM<ԵO=^FE�*��=�F������enR�5�=}��C�3<�3<�^�<�.;�V���><?�����=;����Z�G;�4�<p�+�A�<6��{|�<�F�����ɲ+�-��ǰ+=x����G������<ў�<�"�<�K�<S4뼸�6��.�< �="^�4��8�B<Վ����ǻ�F���c��w����<�����H�埭��R�$�*<	s=���ŻSڬ<�76;�(�H�����<>���X����<=��<����^`�=�����=)���4x�< 0k<ܸh�ǟ�I ?<!�:�������9P�<�{�<1��<��ۼ�퉽�G��w���Ӽ�8ӻ��<a��<Ev���E��a�=/�Y<k.��>Z#��	��0�?��=������(<׸:<d_M�Lա<��	���ڼvt�<ꉏ<��h��DQ<$����T
<��;	����ϼ���?H�f��6.�ʏ)�`!<�s�<��+�C鞼L��<|�T����4�;���<Y!��BZ���4�ю@<*�<��c��0�<����X�E<��ɽ<<Y=�����M<+m>���p�]�:׼�I���N�A�<=Y&�=3LƽkϼҀ��4�x��9Z��!���v��)��=�i=�ԩ�A5�Z�3����<M�=e���ؙ�X@4�@C<�O�<���X5��ي<�X!�U3~�-�i���t�k�����=���=.��<�4���'�=�)=�UxM<Z}�<��{��k��-68;��g�Mm����eA=��`�zX*�rYZ�fm�'��3��g�>3G������=["������Kg���!�}v�;���|=$�p���3=[=�<G�Ƽ��<B�;5v���b�;�(=��E�<����R_<t��<�3��[�<�@�����G��<���=B�����q=BL �"��A$:{8��9��$=DT�ܫ�=�����c@<0�p=��-� �J�;�P=��g<��@=Y^=PĽGC;=�Σ�p-��x}���d��P��������2��<ҩ��B�o�=h�5��;�<*�9�N������<� �:@��<��<�驼���A�d<�.�<9����^��R��gw�=��O<�~��<��Ҽ�e0<z�J;��<�H��]o��n�Խ���=v�H�aL�=�5��5t5�#���}Fm���
��=���=����'�ˬڼ��ڽ����U��7��Z����=z�=�����~�<\�c���ļ��O���g<�GL�C.�;��E=�]��|
�<�1q�#u���<��k�6 ��S��L�=]�A=���Zxh��+A��ҽ3���нeZ�V��Bj=u�=x-��y�=��i�����>�����[9����.�=���<��I����v_�E�f�'���X��z��<[T�*5�=��=��콁�A�^[��0��r���,��փ<��+�f���/l;G=<�޼�H'= 8��T���h�Լ"�Z<�����M�Z��J�;Z�꼡b���2<����f8��0��u�G��S=o%�=vS�L��;n�L=��䉄�3��G��AS����;��a=i9���ż+a�<����`�w^����<����� ̽g�=�p�(��<|<��=��½t��;�K������2��=br����߽��F�׺G��o������w����A�#
;Í<A�<���;Cּ����]>=���Z��K�/��c=�t;o����(�<M�t���d�'м_���vν=��e=�Sҽ�|��o6�3�ѽ���;X������;*Jl��?�<�x�=C}��J�ܼ�T%<Ľ ����N�p����>��=�
=Wh��Ui���}K=�B���Y<�f�/%w<�0�;	�;ɗ��D ��K=���!\A�, =i��b%<����y��<�M7=W���9J�w7��Y%�9���-���2=�����=!v^=o�ν�ծ��4~�3#Ͻz<��`q�	� =���v��;�)�=󕆽s�r��w��.#�h�N���ý�X,�*����4D=<q�=�c������v�e=ħ�P�T�{S���'��<ѫ�>�<s.=��}z�;<<����13�$��<,$�ά޻i=9�=)�����ڼ�D��:*�C�&�ϼ�q��z��;l!)=������#���<}���	�;�S���/<�$PU���=�K�:�� ��l���ϑ��9E��ґ�R&:��ߧ�M�<�<Ą������1�=��=W�t���W��\<��<�v�����=ݷ=�г��怽ɝJ�������;�޽E�n�r�W�<�{�<�w�X٧�����9�\�1k#��ͽ<<Ei�t%����u<��ϼ.v<bO]=�2=�*"�tx�9�b�:V����E�=�ɒ=%晽A?��z�_�4��fs�Pl<��Ig<E�o�"�;�=�&��ެ	=�٠�lV ��I���
��{:���C��V<�c��|��8tg<��x=�5޽9ye�f����qyP���G��z����2=��=׫=K�&=j�'�iA9l�뼐F½����uW='Q�:��cOƼ������>��B	�,�˻=+R�U��;qQU=ˠ����t=�Ž�0G�$9���$������@$��#$=u�;=�͢��Sh��������2*�<#��l���������WY==����O<���;�<�LR�)ͽ��b:J<�C��ۼ�;W�);�$���/=�ݼYƻ�"��/����<7y���sF<=�<ޱ$<Sɕ�I==���;� �$n^��ui�C�;?3@�2��=�|�=�����:<f9ڼ�Q<}E�;���	-=6I#�z�K���N<������<�t��>�<�V ���v��� �1��w��<�>�ֽMv	=0n�����t�j���E=#t=.�����j�s_���޽	¼����^��(ĕ�Y��:~y�<�� � }�=�T��ph�; G�G�<d]��\,�}.�bH=���t��<9�"=��#��F �������
�)�W=g=����/�9��8���'�K�D�����a��o�� 싽~O�:��h��.�<��=8n��T��<��Ƚ�����
�����2q��+�w��C�=�D�=�������w1/�B=�u�<�� =��ͻ�6�_����?�;��F=
� =��<:T6�Y��	/����;H� ��*L</U���O��������:x��H�$ܽB�=��<ǽ>�֎<pvڽ�>�~ۼ�����Ҽ�7R
�-��=�5��k�<�/��Ɛ�z"�<M��HM���*���y�6���ۑ<!<@��<2���e����p�6f�4)�Ϊ��]=V���8���jS=�������A
<9H�;z�ɻY�<�F������ =;JK���5�<��ʻ���b���p�<(W���\�}S=S��=#y=� ��{%�� �^��J��/_��҈�3�D���h<�<�B=�3����;�9�T�=ٷQ=5"����d�Ԝa�1���X�<_ν�=L}�[(K����}�;}wl=�,K�F'�=��\���ɽ�k='�<� =	��<�\<�0=��7;, ޻�Ո�7T!�dc�=.�<}
!������=t-n=�Nk���<9��� �'��<9��� ��o���*�_�P<�J���
�=�T)=�m��ڡ<)����̊�֯��J�����/;��Vxs=G�=�&���^�*{�=������b��Rjƽ�!�G[M���=�R�;9�<�I��v�ؼ�!<�=�H=+c�<D=���e���"����:�`G��a�g�S�ٽ���e�;Wy�<Y|;5���NN�^��IѻrQ�[�w�
�����d[ѻ3ݛ=��P����ƦS<�
�� VJ�尴� ��;�ż�0 =u���>�����������CA�(Lֽ�I���To��������	>�f���R�<j���s^�<7"E��Vx��|�<H'E�v��E;%=�F�<k�$�Q���N=[c����;:)
���9�e3���;'�2�R��<y���M���1�M���ب�_`U�o��<�<w'<�ɼ������1>��nv��A=G^�++�=M���nAQ<#w�������Qq�JKc��7��G$�wI�=
��:By����=�|숼��V�M�Q��0���p�;y��=�v=���t<�/����;���#��פ�</~<�̟=�nּ��<��<��<us�������=mI<���Ny=��ʎ���:�Z�;�Ľ�Lw��Jq�&�B��.���]=Tk=BT���'��~=�����Y�<��d���ɽ�p��eI=�]�<^�W�2��횼F����;1|�1�=~_�^Vܽc�O=�M=<����Я���#���n;X�
��XK�L�N�Tჼhh�<��~��!ۼ�mE���� �(�������Տ��h�<~Q=���<� <=�y��@`��܀�:T�;�O<�%$=j�;
�1=$>��t�&�9�?��+���i��~��Hl��>�:�����=h���<ŭ�����J=�#��,SN=��&��S���R��=�^��粼�.Ǽn��:�;-m�����黼���;z�(��u=2�6�|R	�&)�;2��<�b5��#;KGϼ� �(j�v(�<� =������>�Y<r�=/�F��ٔ4;��===�ӽ�#&<sdI��G��T���ߊ��|���m=|�/�*n<�0x>�3Ȇ�w2�<7��$����Q��V=��=���<q=�24<YEs�c�������t�G�����J ��Mʼ�P�<0��<Dg=����,D����<Ɖ�8NcG<�ȍ���<��=����<��˼���0x��(D�]�g=�<��^�}= =��4��Y{;�f�<=L��5�P�˃\������1����;l�@=*ռ�|���N=�0���-:�����=�x���ܽ���=9ə���M�C�i���A�����.=X֫=�8N:���Β��ʕ��p�=����2��*�^r==tM=��v��ft��ϟ�m٥<6r�;剃�������Bm=y���t�*=]����7������e�<�,<�j=V^X=Y^J�*�8��ң��K=R�4�H�/<�˼��*;q^l=s�������4�<y ���t
�:�������M��4$=��>Qkw��h �ȟ<�%��S�<��"�gL��% �,�<�M?=�=����V;��<�B5�ZD-;�&��=d �n��P��fQ�8���'=�q����<Q��<v�;�.��m;���=}��L�u�(��;�2�D�=��,�3�b�.�=����j�E=�u���F ��4�=���I��g�<o�!�j�X=��=�0t�tA��W���T�I�.=׎��p1A�������;�V]=�ym���ֽ�!=ߋ��t֍<�d��� �:�rw<� r<M��<�Ď�¤�=Gs�7z��6�e��~��v�߼"����F%=��=R=�O�h��M���%=�Aؼq���ż�
9<o�E=c0k;�$c�؜��l��b��[��؆�����
�=-)�<}������`�c�ʓO��jR<L�=�6D���+�<q=FÐ7n =?饽��<j��׺�ó��c��<��<᧢<����x7�Nuc=��87�;�30:�]��x	�<u����=�Cf=�*�	����˽2�:��ڨ=|DԽ4��;���\��=u�<�_Z�Qp��
�佛a�M9�=�l��]�*="̈́��=I\{=�c��\��A�Ҽ���s/;S���5���h-��1�=��]=��Y�շ0�����uN$=*�뽺�(=x���t���T�<���<Z�=��估��ps/=/8����<X;{<X��<�ϻ<w<L�c�m���h�������<�Ｃ�%<ީ<ǈ�=>a�o����##�����az���ļ2��<�N-=[� �7�ɻvG���+<�_�	0d=�M~�_���fH�<��<��E=�K�����ӭ�=�`+�����6W)�ax����2�@(=�O=�B,;� �v����9�b}<��ս�g�����A�<��8��j�< M=9؟<�ؾ�r�B=_��9P�b���;�j�=/i�<굿�+�Z=��!�m2O�P/�(}����G<�>����N=È=k���A�����2�L��<�eܽ 1�PDJ��.<�_�=>͙�4"-�i�)<���F;�T6 ;ܜ���7=�9Z;>J���Gj�<�7�̱�����G/e��<B��="l���z�=Y-�^�[��'=@��<�K!<����eZ<9S=�w[=��Y��̼3����N�Ͳ7=_�漦wR�t���}=&9?<�[Ƽ�����ۑ�C��j��1.�<��*=x'=���;��<�[�up�����߰����< "i�좵;��<o�_=���������W��a<��J�e�%��k5<h�q�=��:@U�<���<0b�����<6 =�)5<�%=�����fC�;=Kid<D��<�/��FK<��Ѽv�=�����=��=O���Jڻ����8�=��ݽ�)m��k��텽�
�=�E����<�a}=��7�g-=s���"������V��>2>BB���������-�.�e��> �[��dN�U��� � �-��-=.��=R!��}�~���¼;B��:W=1�=80��4={z��L�<���k��<���<L�=����w>=�[�<<9C�������<��=���),1=u���!{����=p/�;0�<��y�1(u�����8�ϽG^=� =���9*=�A�;��0��F��bt����w�����;k?��=T��=�ش<�E�7	��U!=��L=o�̽ԼB�Ͻ��;�铼_��n�,�5�<sGR��+>�	=|�q��k0���E�UF����	=�q�=�ռm6�;y@�������w���"��ͨ��� �<��lF;!�8��	��p����,u���.=��;�@�;�0�<Z�=���;�t��ʓ<,t��/Z;a��<�<�>�<� <ޙƼ���z[��B9�߲=�eX���0�&��|x�<�)�_�=�� =@���<���<��;�~8=d�����2�'�<G�޼[U=��*��~=����%u�<���<#��9�K=h7=�ü�R��`-=�1Ǽ�B>�2�=�o�{3����7=�f8<�T<=��<E���gH�:������_V�d��9r�S=���<���S��=ݪ|��� >�ڣ���R<C��0t�1}=j�j<��i=񃰽��\=�s��(� �<#T�;��-=1�ƼoiP=�E�Ճ=�ˉ��R=e6=�F���f;�@<I�ڽ�0Y�}�9���:�jaû��=)\�<Jj��i;��ƼH�:<��^=Y��<dF��ټ���(<�}L��'<f2�<�/���;6=)��t��	=b�<��<�J"�����?�d�Y��^�O��o��0=��=�
�=󤚽��"�p���q_=�7��B�s�g�D�魻Ew<d"�=w��<�XH��)<=,ӈ�;=X�	��;��Z���j=9��<�ѽ��;�aB<0���5�%�-;J,=]쿻���ü�n�)��;*½-��<��;}�7<\�_$�6Z�=�����ؤ�S�i����c�h+}�@<?<<ǽMnm=ƴW< \>��K���ڻ�=����1V= wq�E��<r��<x�s<:P2<��=_��<<(�⁷<�.�<U��:�ɼy��t^�܅1=1<	�<�:A��X3^<:)`��%�E�
>�z"� �>h׃���&�-�f����q('�ɵ�����3���MB=^�=�.ֽv-�<w6����t�<�}�����=�Q=�4��N=iÑ����<�+���Ӽ�vK��EK=C�ۼ������=��j��j�8u�;Y�<��W�0�V=g�����=3��&�\���&=����:T� {�</R=0N��3 =
à=���<�ݝ��2=��<G�<�d.�ʹ��#�>���A��5'=x,=Z竽ȯ�;�H�/�)��=����p��}��=A�=ѱ=�Rp�1t���H�!� ���G�S��b3�u��=�k�=d3�=R^���=�q��m��
��7=�#W=�ʎ��$�<ɼt�I���:�^;n�8�ݼ	?�����=��=a'	=�I��Sd3��d��=p!�D�м	�Ͻx�����>�t<)��=�>b<�X���L���"Ӽ��\�A�ԼFI-��`=��=3O���F��\���󽫲�<����1���y����=��=�N'=�z�pê:)h۽�A&��&�<���X��9� =���<o]=f��<��;���LS�<��)�R�H������<lw�<����~��<�	 <><~��''�I[	�
9[�?�9��|>hA>z�{�#����ڜ=�0��#¼�B)��Խ��!����=At0=�~�=���fﺼ�����мl��<���xa� M=	;J=�s��!˂���	=�p�6��� ���G�A���8��=���=ɑA=�ݽ��<o����ü�c�<x�μZ�<��<*l=Lg�<&����;���	*�.�<R���N:�= m=d&�+(%��ѣ���=�[=X�]���.�m�������wx�n�=�=�%��w��]�"<*���	�����c˽I'ݽ��@=��=��Ҽ �	=��=_3g�z,V���= 鯽��,>qa<B���FB=�b=�N.���?��D��s|=��<J�=������ļ�6���QS=�B�<�g�Y�8��2H��m򽻐����=0~=�ۼ��ܼ��l�~��|�sX�;w�[���=�`��ULʻ�ռ_����H=5/��#��ƚ|<L�м|U���=��>=W����(ɻ绪<,:���7������>k�=��=��<�Ǵ�i�I�����q5��W���V|�/,�;���<�7=- ����z9=��<�2��b����!;����
�c^�<`��<�	9<�뼃M'<^��h�;����	�>�d���R3>4��;�R��9���Wm=�G��Br���^���C��4��?
=�� >��@�����6�=�2��$k4<ʥ���ز�Z$�o�f=�Z�=[��<8½����;�(S�3G������{�ϽA߼=|��=��=~�+�ޘ�;��%�<��N�<B��;�B=W]=O��<�A;36
�d,�e��ma�^L�=w��=K��a�;g6k� ��<��xz��I��ݴ�� ����x�=8�=!��<$�;��%�������f��#�=.ҼWĸ=���Al��U����eN=Ɣ�����_:� ?��={=躕<r�<a���@��<��<䘌�"�!�f���s��˿:O��<.=?TH=�c=�
c<�7�d^�w):��e��RZһ���=�˄�4� �s�
�U��;�յ��\=�H+��K=�t���l=��`=�ʰ���:�;6Q=y*3=��Ƚe��4�ý�6=�B�=�P�<WwӼB��.^�<�%���Խ��S��x��q��m�7=�e<sJ���u�� t`9t޻�?���J<!��<l��<�����C�<B9�<Y�K�ā��'���Ǻ�hλ��
=��=��:/0�E�i<���<����|��X�}��|��~ѽ1P�<���;��=�X*����;/o=V߼����w�+u=N���æs=���<�[�P|�&�,���~�D���Y�k^���<���=詂>Z��U�a����=2����0��е�h���S�=T�{;�H�7M'��}A=n��x,��1����n���K=�}=5:��̱=<�6�4>7�D@T=��<����|z��� <f���!��qW=d8�UlA=ms�<KjW�.5��&[=��¼W�=�U=��7�� ��u��=�	��:�j_D�=� ���s=� �9��"��iz�<�<.��<�#�l9��.�'�����;&��_S��f��a9q�z���dՊ=��='�(=:�����:�KX;%�2���y=; ���u�;�(;{ς��V�K�E=��<s�ֻ�М���]�;E�=d�$�o�==4����ʽ�K=� =�̕�	��<)���=�-
:|��=}���Ǽ?�@��}�=z�U�ut�:L�ټ἟=���;�5G=�{�����o^=ݤ`;fxƻ�U"=v�������0��6#=�T��G��<񗜼<=��<���<�t����:?��d�s"ټ=c8�[��I!8=k�h��c�9oi\<�٦;w�<,k<�C<7�̽�iX=�	=�u&=�;!=5�伽��A�<v:�=^)�����9����=�ў<��R;Sϻ~g�;���'ӽ�$=��<:H���~w=mK#�O�>'8�p���t��v�5<<K��<3=��V�I;�;X�K�=��[=Af<�`�7֛<h��;0��=6�O��;��=��)�`��;[�G^���@=���� �b���y<F6H<4�U���,�ޞ�9c4�<�#��X���A=�<]��J�9�f_��ޜ�d@�<�۵� ���G��Mt>=��a�������D��h�Rj��b���<��=p�:6�=�aE=��X�qn��;K$=�qR�
(=/�<�s=��>@ӽ}[��;T=0�:�~�M=�-���M;|��\=����LW�<5]=�^��y��<s���S!e<��=���=���<Yu;�4�~< e=���\�����¼ޛ	����=��=��ռk~�<K���De�=#P���,v<֒T�,%�=�%�<q6T='�<�C��ꐡ=�<�;U�[<3&0��c�[�p�F���k��<:g=J�=�Ə�}��;�"=���6��%��L弦^�<� ��9:e�<���<q ۼ�hD��<7=��=:X�ļ���F�=)��<,�<�1�=t0~���u=A���2�Y8��2#�ß<�I�<���0k(="n�<9I���g���L<slӻECk�ޅ�=9�5�j�ս�<�=U����v�>�ٽm���w��'�<������<x�T=:7��Й�Q莽#,u<Ty=�&�=�8�ZS�=��˽�$�����ݬc=�h�=QF�<�G����=U�-<?�=�k����	��M=G/�<J��=@���L�߸qy���m���,����$=gL?=s!���%��D��KQ��zO=Q�=3Ņ<����=�=��c<�<�M�=���;o�3���!��C=�V�=�e��NE�:��h�!�:��t�=|��ª<�+N�kH�D��=k����G½�a=s���B�=�O�Ϭb=/u㼤�6=4� ��`=�O�=H�Ͻ���<�ϼ��߼~U�={�.=�@��eû[��<��=����"<����0�庎�G<o��=���<&q�;�b<�9O�<�5���<�М��i�<Yޣ<�Բ=]ut=�@=�Ad=��M<�s�<O�˼�U=�:v�����ڽ7ǔ��T���I/<D=�=FƼ�$=��
�����\'=�B<g��� �<8/=���<�ּ:�"��C���=���<(ؽ=�N=�=<�.Ӛ=~���W��=����l�����֐A=��<���<?����ʘ=�K��NC=&����A=͈�VN�=��^=n{?����6�I>�_Y�B�;>�q�,8�=m����FX��׬��V=���<�3ǽ��=`3�U�d=o�K=X�X=�1�;,[�<���=��䢉=����=�DE=`��;�N<�Z=T�=�	�����������<d.�=�[�q�U=���=�E�;�����<�ʆ=�ݚ�\���q�_=I;�;e�= �=��|=��=%�-;�!�<b��\�	�uY�=�Dw��T��e풽��%=�4
=/9��P;U��=::�=\k=0�^��q=;$��iC���o�=H�"���ս�q+<*H=N �=�Y<�˂�Wԋ<��$�C�Z<g�=����cb�]R����p=%80���<'��Dǻ|n�<GeK=.�<�E=RR= ��<	*Ž׼<lPO<�ϼ�>�j㲼�A=���<ݬ=�ғ<z6D=I#=�I�=��;�X=.�<�,�<;�=��M�	G�=�Z���P<ý��3Z=ᜁ<��Z�d'=����\�x���
;8%���<�U�_%=�5Y���M�L��<L�;��=<�<d:}����<���R�>?V�x�ٽ��>���k?�=�����g�<�7�:�\���=�d`<��;�*�=��<G�;wݥ��\�;�!����=��=�ټ���ш=��>�{��<	z8='}�<��h�J��OH3����<��<A۽c5�<��e=���1 �-GJ=S|�<�=��H=���;���;a��D37=dμ���<�d���3��A�fȌ=�t>��=�r�=��t<���<�`��Hǣ=�Ը<2��$^��O7�<2�&;-��T��='--����<ه�;^0t��sG=`���i»�8=EC =}��<%�����d��a��� <bҽ'ӽLc>3�=�=�C!�TiV<�����<�w����=Q��:���Q�7�Ȳ���;�A�={�<�����=@@�<'v��P�׼�ں=�'�<s�)�6�<�7K��C���)=t���&�� e�=�=>Ɗ=�����8����ֽ���Տ���U�<擄<KOʼ���=�LJ����qz��>��I|�*��<�d�������=e9�<h^�=
���a;�à�u��<����D�����Z=��^<��:�2�:)��y�׻���<�a:=-�A=��<ZՎ��l���4='�=��W�!<���֖�.?`��0���=�i�<�
�=/b�m~e=�A�T�)�C��<I:���夽M�i=�=��^<W� �h='�\ʗ�nc��(i��`�<4@�;to��u=��μ�������;DϺ�?��;��;����0�<�l7�Yl=}�	��G�=g<��;WF���7X=,�������_�����Է��N=s�˻%b��u=��fK���>JA��f(��)�D�{D�Gx=�O3<���<�V<f_�=��"�G��r�u=��\=�_=�!Ž�؉�<j������R;�z,�lĕ����=�)�=�G=�n?�6�=�x���=���0=^���8�=��n�XW��\=(�m�=4���N��Jy�;�Ȍ<9t,=찼�
���!a�����٪�=�r�t�����N��1����:�����=݊$=�Ӑ=�UQ�~5��D�潀-n�%d�vP(����=`�<נ~����<Z	�=�w<޿��R/J�s�g���#N����=�l�<C���V�����;��#�Ľ7
2<n�/:�َ=�yں(n;��&`=D�=�0=w��n�=��+<�ݼ,����D��;	{k<;�#�%��%*g��.������ۼ����	=GM����<�r�;��Ἂ�o><a<�=5A�<Ҋ�=4!s����I0�����=��h����d�4Jw��!�F.i����<�aw=b��;T�f�t=���ݕ`<��ۼ��H<�ɀ�j��;�<�=U��;�B��� ��F�u���]���{"�B��ֿ�=�f=��ۻ�~��K���1M�r�5����i�R=�#��� �<
�<f|�;!�6� =�<�u��a#=Cef��eM=0�=�Ů�sI%<|@�p�}=�Qͼ'!ּaX��"}�e	!���[��=��=IX�=� �ȡϼ�\8�ʍ{<���+T�<�m=�1=�&,����p�=??�<bc7������o�u �<��=�X��j�,�X�n�(=,�8���3��-ټ6`r:�\�<>@R�������;>���h�g=�Ŕ��)+=n�<h᛽�f=�.=e X=uu����Ż޴%=���B���f!;�N����!��Mg��/�;��.�zK����y=[���.3��/�<�����zS=���<�u��M��:Q�D�m��<���	�j���1������ף���'=o�̼0)� jF�_2=,��m��pϔ��F�;�J��}�; ����<[<//#��u<�c�оn�f3��^�<�8<=)�g	��U�<�|��p(�:ԗ=��z��Ͻ���02	=�j<
d�1��M$=9�W<(��<_�3;���;���w�;`#;�Q���E�E*�=D�
�:=1���Ƚ%���7�5I&�=+Z�MsT>����b���>n�
�`�һ�ǥ�RV�;i�=��
�-"������4�;��;<�v��{��<F^O<U,�X����+��!<=eA0���>�-:���;xK7ý���O�<TA<7X'��&T=��s�;#9��=�]ʼ�!�'�(�(�<7�=TPi�~m���m����=Q�ܼ{��$r����f���<`(�=|V���X� �+��5=�=��<��v<(F���|<�d=�c�֕~<_뼴uC=�<�B������w�<Mv<�U���;�G=�)#<fv����<̕�; �`�m����8"����<ðn��sn:����,�/=��<4�=������X,�FE=�=��M�7��M���~=�,=�H=|�q1��$~a��U]=�M���
�l}Y��pS=���A1�<�^<H����p����=�K�ܤݼ�^���=H��<���=j� =�;�;_����Tʻ�v=`/|<�򼼡���'�����~����	�Z�2=d�y��g�<�5,�Ħ<�w3=e��<y λ����h�/=\�޼
���E�O�׼�<ګ=4��a ������*�8�=�5=�Z�juJ���n:�5��xmw����Q'<�===����#(���=�d(>/#��TmսS�1����n����x8<�@�<i9˽����ɰ�/�=ET<��F:<�m�*{��aT�<Yi���}��Dg�=��<�M���W�Y��=�y�g�,���t�g��'@1=�F�=t�� 6;���s9I�:a ��ص�u0�;%=�gb���n���/U�=K-�<��<an<��3��h��H�<��n��=�<F��;�_����=b�<�ӡ<��5<z�;�x�:?�����6����=�~Ի�P;��F=�s���<2y��Z=W�X��<�V���>�=�ʥ�t/��M ���<�ע�f��;��=qTq���=����K�<�5��C@�=���<��;��=9�ֻwjM�;ON=( ��Q;�<���=���SW����<���<"lt���l���>�m��;��h=U >M����OȠ� :��>�9�X��<�6=R�a����E�<�{���n�<9��<z��.�ۼ�|=d�u���<Ѽּ-<	��<w�'<�l��=�<�@�@�(�1z<<�,�<�}�<�v =�=É�=���>'8�A�#<�IS���8=Zk�P��b�=Ւv<��_�5@����(��Ӱ=��E���"<���;����v���	��]>������a?#>����@W>��	�1������'=�ݑ��o=�ƺ=j�<��<�)m�l�/=�=~y=z(��b�>�LA��\�k�FЖ=W+�=��&�YC���A<����տ�=%�����w���;��5<�*U=OE����ڼ��<�>�;1i����;=��`=���	=��I��P4<�kN=�͔=�E��G'���}<�kλI�ｆɈ��J=�;�-��<�.v=��μ)q\<0D;
��=�s�Xh���Pϼ"����W4<s�=?==�!>�U8��3��f=[��<�*=J��;���̡�	�<]��D8=��<��w��=c�c�Dp��t\�=���=��7=,�y<	��<�pc<��v��Iq=��Q<��%�W�<N�>]|<j��<@�!=^�=*�Y�u=/$�S�ݼ�9E=���=��*��<i��=�����\��Gr�Xk�<�;��/�^��;=񆼔�<j"��v�;�L�;���;�<7�9��wx�5��<�y�<���<��#=d�<��A��wa<��<���������fW�d=㙱=�wļ$���;2�="�۽H����F�6�b<�17�|��Y���Q=0h=Oj�<V��W�;�m�[�v�j�<b�Ἃ}�=���<m<�ɺ4>H�g�i(>�=ҽ���C��<�ѩ�g*h�F=�}=5�w�7Y����<P�B�0�)�F]>���F��=��g��[K< �<6l��ܱ=Z;�;	,���&=c=]�,>�/��\��H]��W
=��=�(�(�e���>͇G;!{	��:�<Zf�=~���_��<3�i<�a%�T�d=��=�<aP��Z�<센�n���5��6��5�B=����<r�=&�5=��2�ڴa<�`=1Z=?&"�x)�=���63�xm�=6�j�ݱ��PQ�<^>9>BT�=�J������ǻ�@�;PE��C�<+z�<[�/�z>=<	�;Q���n	�y��=��p=!{�<�D�<�@���ѻ��!'t;�T��MM�M�u=��8=pה�QI�<�G���=�_�F=jfܼ5ѩ���c<{�=�,=ft�=�J%�&��a.�=���=b�<�����AѼU�
�`/��{;�ܼ��=�S�<ì�;Ze�:�x�a5��u��\��܏��L�<�����w���9O;k�;�ꕾ<��<<��<<��ӛ,>�l����>ٿ��tR�����+;�*ӽM��<oЀ�@��;��<��ef=K��eU��	3P�}B =�>	;�=V�8<J����!�98=g�ǽ�+���|�G�λۢ=Mwo�����A�<>��<͏�L7�=�|��s��KF=	�[=gW�<��y��o.����;nO"�Ic��ִ(=��3��F��p�=�	F;ګ��_�=J*�<��m���7>�m{=��<�(���=�Ec=�-�;.�ڰ�=�����S�<&�j�r��b=*�=�,��Vv<\���E����|�=rd�=��=���<=�.<ߛ�z/�9���L3�DᨽU<>D�R=�e�<7)�_�<������<������=�h<���d�J�M3�rRռ�uh:*�0<�<n�3���޼zv<�ڈ:@;�;�I�<걹�
�<�8�<�\M��6�<�J��^:!�b=���=�H�?�&<�s����<3�����;+c��܌Q��fv=�!˻�!I��&=	uB��0<��D��k��Nۄ��5�=l���!"�=�Ͻ�o�=������;9{���É;�D(;#�E=�R���m/=@s�<�Z��d�W���[(=��<���,�����4<��X<T��,%<�=o"'���V�𱘾��½��=d9=��=ݡE�yO�=q�����p��=� ���<ZV�=Me���J=����,��=��׽+�q=I��iA�<ƝK;`oQ���g��Y���>������<�H�K,��J8	����9�VT��iK=��h�����LT=T�����<�s��T�V���<σ�<�?-�4MB�"�i=l����<�ĵ<�U��N�=+�r;�8%����Q�=���<����w��=ź=���<�ƻ"�輯M�;���=ɑ�̎��*�&=Dh�<󎃽�ނ�u�����ܽ&�=��=��<H����=������].��ҥ=��=d �=�d����(��������R`�~8;]7�<�X=H��=��мV6�$�)��8�<�9`��C��܏��'���~����;��3�=�]R=�=�佖��Z˒�"��C@����N=_-��8�:��-<�&!�	��</B*<$�뽊�	��㍼���˃;��P�=�=�b:mf���6=��|��k����k��<w;K ;�-=�~���Z=��S=lJ༝39��؁=X��<~�z�Ės�$��7Wb��F���Ur�c�B���	��i=l���\�<�1;A��&�*=Mn�<��=��7<��:���L<��ռ���==,�=�E�=8I��K���J�<ӈ��ӹ�M�������B���V�r�z�t=!ՠ=���;s�v��m=�í�!�.=�*<�h}<ƣQ��=�����W�<�_����=�|J�Q�<���y�ఀ;�ǽn�=g�<{�<�r��� �<Z�e�C=1=[H�<��׼ګ޹�3T<{8=pn��D�;�u	=�����"<x�����=[�_=�9���H��Q���d\��������
\�����3��Cp��\�<C��=x�=��۽���=�����;�٠���&=O��<��<a:���妽�
�=�i�/z���j��II��&�<�P�<�Bu<�竼�a9�U��<"=Ķʺp#��\V�,���N�1�m��x`=�0g���D�������<?"E�i�����I=ݗ�=WA;��ɼT������=q
�<��%��S=����iȝ<�G=��B=�3�9�����=0m�j�Լ5�����5�]=\����M<|���U<���=�G���)���_����������]=*�=���0l����<	"<�>��ê�<�<�J��=&]=$ ��*�$��
=�һ�!-<�����vμ! ���J<no��Y���Ҽ��-="0/�x`�<搴��1�	=�*���|;��R<��&��ݭ=�<�h_���U~��؇^=(��<pl=�Lӽ+������=�$���~)�D��x戾M�)��mݽ�e���@>�+A=<�+�A\<>�U��+�9����!-=B�=�-e�����}�:���<B�S�Ϋ�eK+�^Đ9���<���-��;=սO=���	��f=�<I��<�Fg��6�;*!�R��#�<��$��9)=RZ=�=j��<�*[�@�=�;(B=CH��xZ�H�=R�������Qռ��x<�(X=^Χ������=�Eh��[�=�����M=:����0��<��M�0��<���<��n�s�=���<��;oe�����<��p���q��<�z<�T2=�3.���:�� �15��=<5�'��vԻ�	�曎=ц����=/4��Y�<I��=%�*�7rA<�޾��{�<)ͺ�<��=s =8�k=oΊ���μ�/=H����ڽ�q�<��ؽ*l�=����-;��+<ƪ��g�����<�,<z��<�����H�<�LP=��)��$ż�c���<)����z�J;��;<�|=m(�a�N<�==Wb~���꼕�C�u��<��J���2�FN�<_g<�����a�<���L���I�!�qR�=�샻y�<tI�<�j�=vSD;�鵼�޼�hN��m�R��<�\+�\��='����$��(��=Lƿ��g6����=�!�`}>�D-;?]�=26]��E;֪����?_�;n �ެ�=~������=��"�H!�=ua���=,-s������н 
�=I��<V��n"˽�>�kλ�9�=O5��ֽ݊�Տ; W�=�2I=-��j����o�<�o��:1q��޼�+F=�᛽%=ƃl��=Z&�<�8����=����+F]=�"=���f&@�^I���iV�
m�;�<���;
/<z���'�<����Z&<�>�<����,�:�l�=|�d�O;��S<5`�����=�'ɻ޳�= �D�8���",�=.=�>�<��1<>�;=����Gu���?�p;���<�M=t�t��䛼&�1=��b<�v�������<{<zCu���=�� =�*w<ɮ{<d
�=zm\�@aE=h�5�(t�<��<~�=j�ѻ��;C𞻸��<^�;��3�%=5ɽ��;��n0=�:��+B��u<�c><�J�<M�;�w�]=Q�4��"=#Ȫ<}��:�=�b�"�����e\Z<|O<��=/�%=�w����<g2�N�)� �	<z�D�C<�l�@L�@:=�m�Qj���=��Q���5=�����v�������M��_��o1>�
��x����3>�R<[�>";�`״=B��u��Xr���޻ ��<0���4�< �}���!�	�*�=~l�A�=� ۽"�5�[��0�)>U��=��=Q��8�����ꔒ<���<%�?�oCQ<Q:�<�5;=o�/�m�2��q>bl��UÄ��V:=���<U0���r;�%�;�㼃J=��E=3{�<y�L=01"��_��(轵=��Ҽ�!��������=��R<�ǘ=�v�<��o=X����=ħ&��̖���T��cY=�_;�+c>�Q���;�Eό<(�h�� ��fn��*q�Z[½m76�v�j<8��Ϟ�T�Z� =W�2;����K�����=[};
5=1�4�,���;�E���$=�r=ZN��OK��1� >&�y=9�M<[6�0T=6G�s�=�,�����<9�<��(=�0~�O.@=N�.��UG=����r�a"�<��[��_.���.=� '��\d��>��<��<�k<)o���Ű<�����=+h�y�	�|�>���������_���H��ά��Ļ;*<�\;;��[=�Q�=n��<!i>㽷���G<Hdn�N��}o�<0�%�o�!���C<x�=��<�T���:���;@E1�uZ�-��m�R>'s�j�Ӽ��=W4�����=g��-8���r;�V0=��O<�m�H!=q^_��H�=ӌ���4���6���=.���!>���<����C���R��~>z�P�/	ԽR�>	#�Ur>�����<SXw�c>ѡ2=��m��8#�mE�=�c<r2^=�L?�z�;F�\;����M�' k<}�<�f�=���=�����C��tQb=ٴ9�97a=	����<���<Ky=�g?=Z����?���A��=�Ǽ�ae���}���rP������>�<��/��*��=�J>f3<>��9�u�_�;��"�=E���<���<���C~�=3߻�=g;<�úٻ�=Jl�=TV�<��,���/�a���*;�J���<�M	�<��O=��E��ř<�A��<�/Ľ�	�=�ꋼsK�<��;�^�y=��k=LUw=胵�8�Q=��m="C���TD�؂t�Nz�;*��:<>X�< �-�!:=]�/�o�<O�.���=��w����<���<2Y�� k	��i6=w=k���u =�M��X�9܇���P?�<��=%� �F,���}>n�߽\�ɼ@� ��W�=�/@�q�=��;��C=	|�<�y�=O����6� �;2�v�w%2��'=���x�\�=5Z=�4�=�!�T>M����P�ȼ�q�<�0ɺXN���B<�\�y����4=#-�9��$w]��O�=E+%=HrM=���<���=�&=�Q����f��I��@�p���=dNͼPxm=�v�=�S�:�1����>����["�>���S�����<d ��-�|<�}1=��(�:=��D��.��	�>�T<]3�<d�f�#�$���I< �^=��= y���Ҏ�!�><o�z<+��V{�;".�@->r#�=cS=��޽t�=������ =�$��#�=[��Ľ.���YM�<��=BT���t�=$�I=�S��l0ͼ�]��V9�"=���<�/��?�˻kyM��C�`�-�ᓽȼ)���i=����<��"�<կ���J��=��i;�kG<8x���<��=�	<D�<�Ɩ<jG<�)�*R�� �<E��=���=�lE�􃽄Z;8臽�[=�P<��<-Yr��=bw<\6;GQ6<�o�y�<-5����満yB�%�����=K�;4�d<,H���&=�6=y�ͽ9;;��xR��w_�&7W>p�>���<<"˽s��;^<e�R4����O����i$�]��=&���rּ%��<���=i<��ɹ���s�DcD=��7<A�>��!��m����[=���<�G�<��p��O��]hF��?<`=U �`�C<dh=<��;�����<t�?;��g<3y=��q���Q�i�R=�+��q
���R:=;.�<	R5>�B�<}\E�xN��5n�=uI =R���>��<{o)�.�D�4LO�Y��:�ә=d��<R7�<�2<������<�o����������9���=�E<ºͼ��=��ӽƋ ������<�j=���=;K6�ѩ�N�Z=�
һ���E�U��<�b�<���: �g��vջD��ȋ�=o=y�$6��Y<$���6��Ų+���=l#�<c�N=�Z�z��=�g�1��vĽc~�<-�;k�Y=U�n<U������=��@<��ǽ̍$�䣔�&�<��J���3���=���W��_�[=�qν���s���,=�sS���<�)�_U�<�m�=�8
�(���v>�M�+=�Gt���\���%��I�5;<����T��=��)��x����:�������30e<?��<��1=Mu^<��U��o ����<n�=�ؿ=�]��@o�?��=����7��*웽B����X�ʬY=p�
>O�}=������<���`��:=g�G����
�ɼ ���!�=���;�?u<����@��aƽ�����ѽ[=Uϊ=#�=s���˭�<=���T�����<�x=C����|Ѽ�p���fuK�j��;��2=��#5Ż��=��$=R����;����r�v剽�82��>��UԽ���=�ݽ9<�=J��=e��;!)��'��=�q �=n��,aֽu�<�џ<L8�3ꗼN�Ͻ�9�=D�9=�}�s�X=�{8�V������j=�u�;�7���<���=�G�;ˋ=��O�`;aX��߲=zD=��<���(<��<�:�����/ ?=�ك=#v ��{�=���p=�@:<1֝�r䱼�����O-�9��<�{���ż��ٽ��b=�!�<�R��u�<ڌ�d�h=������0<����9�5�'B�=��k=����巗�"<�� �6�^�ּQ, �)���`����=����ۻ��S��$;��f�<���<��
�L�t=�e��դ��s��s��<��{��?z���ݼ�3=�r��<�;{z�<.|=���<���޶<����=
�+=��ɼJw��A�`<H��>�Q9)�k�����Eh��{=�B��m`�=��<���5s�<��=�[����I��;�`�W�-��fV=S��=�u?=Y$z�d�>�a��o�<M�ʽ��y=���<&ֿ���� �P�]J�=އ�<hY���(�L�D��f����3<�Q�=o�H=�'��Y�"<�{ݼU^=���r.=�"��)�i<V�<=���H/�"U=O���� m��"CB<d<ṹs<�����нQ<s=� =h�<��yM������/K< ��Ď��
�`&���;���~H=��4�hf2�.Ke���ȼ����rc=�Gj�c���o<+�	=�X
=k��o��"a��0����p�<�̬�g����ԽV�;e4��K�Z=o咼0i?�ٶ��ف>�eh=�G���1=�jG=�o��(�Z=�,R=(�a=�p��ۏ�˖<��L���K=�0����<|��<������8��:��G��&]9<�#���+�=pe=H�<?B	�D1=i��=����B����0j<��UO*�4*�<�����<$@�<�h���ѝ��G�WBj�\ü���>�
=k���yo�<���<���<���0�;C�S<�i<jfݺ^�˼��<.�F<���;�G�<��Y�'�=&>��s���	o�T1=l	���>�b�B��X�<����U�<fvb��J�5Q<�ޏ;L�����2�=
����e�>׷E�L�'>�� �gè�	�k�t2d=�Y
�Hfp��@�$8�yiS=Oߕ�?C�=2�M����@wQ����<�F�1�=�����O=܍�=:!k��� ��/��~�6���<������U��m�<�=�q=�R���T�<`]�;D���E���� ����;&� �8�¼4��b�j=}9�=ഉ��ԉ;�N̼%L�<l��;c�˼ۂ0��o�<GI�<ٓ�<isG=6�<��A=�;<f<?����ޏ;�Q伵Q��G��0=������=�3�d}�;=�V=E~H<���U���^�<���=vNi����<��=�m���##=G�2�D�=�vf���^<qK=ZX�=yl�=@�9=x�ؽ��=�V2�4*�=o��A(4���b<�7�=���<�k�=B�7���=��	�=��~�5��*߫���t<r���Cܼ�Ǽ'p�` &��`�<Eq�D)�;��K<��V��v�����'��:%����ɼjt7�����d�"=wm�<4߼���<�PӺ =�cúz˺�R��S�E<��=�ͼ5�=�:��a{����jPǼ��<�𒽡x��{�]<"�T��ʄ����}B�ݠ�=����b�<�����	;�A=�"��-�=h)��9ᕾ
7>�-=!P>&>����=��F��ݛ=�k.��.u=l����+�<4t���˳=T�8#�2�c�^�X>XJ��B6����C>'��=�k�=s�*<���l�ļ���<�};����<Lo�k�=��=����R�=���<<)'<&gͼ�d����g�������3#��V<Rݼ-�)=��*=�����<?O���_;��F�A�H;�I��N�<�u=��L=y�-=�%�=�.S�$��<�,����,=��������5Ib>�~=DL�=����j�?�e�j=�L��e׽?�E��K<a�=���=��<U�
��j3=7`��v�?<��<�`�Z_=�k{=g��l*�0�p�=�\Q� `z�%1a;{�U=J["=��=13�io�<�?ɽ��w=ԍ�<ź��m����e��7��=�q<|P�;�=����Ҽ%�
�����oݼ��t=D���WB����<�ڙ���<�!=l�����U|�M폻 ��<�Fj<SAʼ�b�<$c=�6T�.3��!��� ��9�^$=x&�<�֟=�=O+=䯚=�o�~c��ݽ��B��;cؼc"�<��<�(�=�</�a�Sƞ�T���R����
�eV>xWW=�@ѽ�n�;4<��=G���k̞����<�R�=(��1���f���s=�^����=�Ο��I����	��n>П�=d&��P< cO=H��=�<�]G���G��_?��B>���<�>F���N�D>C񝽰U>}�h����П=,۪=Ƞ����=ĀϽ�Z�:+���w�<��J�g=�;�;�=�M �*.�=���<�h��biE���<<�P<�����#���Q�b~�<q�=��=�d�2���`�<�~Y<��J����NH�=h�7<-�ͽ|E<>���;7=_��<�<�c�?����9�=	=<%�Q�%<��,�e�%=�V̽�B2=�(��f�2����>=�g�T�F</Yk���P<��<a�b�26�_�R;���=�o~�2<��=ܳm��§=	�˽u��=Ib���ƻO/�<�NG�������=]�!=8y��(ˁ�)���DY���v<B ����<}Ɛ:�~����3<�����5=���<Y�<�δ<�8=����"�Um��s�o<��<�￻mo=g��;l�}�Z �=�T���乼s�|=?V>��C�Y��<)�1����ɢ=#��{\��"=��=��^����[��<�X(�@:�=p�=�<=�=�E�=�>{�4�����`Լ��ǽ6����|�=C.<c�;Vڴ<d����ؙ=a�~��dX=Z�Z<���<x���e�=)X=�d=���:�	6���	�h0��D���w];4׼B>2��=�iV��J="J�>Ǡ���?�>�
��6����1=����޼��=�*;��d��I�� +g=�:f�x�6���B;��нFD��qG<'��=>�=�
����2=-�<o�'����<A�������R��E��= �<���.�9�8U=��ὼ0�;�u꼞7I=����Bj�X�ռ�<6<�>�<zT��Eq<��-=�&s����QXf<�9���=�N<�j<�,�;G����.f�KF5�Ev�;J=)��=2��;(>�4�v�P)=���~봻�����Gɻ>���j��F�=��%�Ta=/U=#����Uf<2�����3�
#�����=�$=������o��ܧ=���������+�<	���Pf��q�<�:�.�U=+]�l�f���<q޻�a��x<0�$=�W(�B�*�N����˾<�t�E^�<;����7N�.�޽�ͽ�T@I>�2=��u�����\�<�/�>#Z=/�Ҽ݆r��<��,=8���FP̼wl�^�<A�༫�=��ؽ�b*=bT=��=]�����<���>j�2��s@���ּ�����<gi�����=E��<̐9=9t��g<���:�w"���g=P�E=��������1=�55ۼ�c�Jm*<e�0>|r<��&�a�=��*=�x�=2�H�#3}=�#=�`����漣������<�=KF0��^e���<��w��J=��0���п����u�>v��#�����=Ӌ����������;�x=�1;�^�c�S,$�b��=LG�=_����� C�E[,<o�n<�h�������k=��.�a=AvǼw�����ӽM��W�E��D >��⽔�F�ž�=�1F�77D�W$��������r=�<�;�<�
�� ��=��=n�����T�Hcq�Ȱ��2~p�L�(=���=o��.W꼅gc=����-W1���ؽ֮1�;Ǻ;��a<!Q�:v���'>����p���X=^7�<qL=F�
=�Y�d��<��<�TP=�;{�T���c���\<B0=�	;0�1=���E;:������[�<�������S�a<Y��=7[�O���mOj�G�> <�� i��3��U%�n�ｬǧ<[��=�=ݼT� �h��=�A���f�<�4���[μû�g��̊��R6=LW^�ǐ=k�=`?�:��ֽ�I�%܉�Z���2oO=a���A�G��ɪ=_���(������������r�N�4���~ ����<���9��;&+n���1��X=a4H=���5��G���U{�������V=�)ʼpk<�T��Mׇ�[Y����=� ��5�a��
�=��¶��2��<Za-=|�t���}��<��K2=S@�+<�V��~��<��)�run�����;�'=^�B�Jܱ�zr��y����̇<���<���<D-L�o����3�;����8������A=F�<��%��B��� �F/�=�X�<'�ڰ�*O��Go���A<#aɽPM�����vm�=��=%�b�n�]�����u$=dSm�ʛ�R�;�7����=g�<1�b<x�;`3V���V< �=�}������CEL��i�;�����	=	�+=ްżΓ�;�P=�Xc<��?=*l=��7��Y<.����|:�}���u=AN���ͻy]����	�y`�;5��T��������=�-��<�<狀�� 8<ݑ�<�1��<64$�PŶ��Gy���R=��=�~=�d��)Vh�С�)�=B2��`�E�����㾇�پ {�=��a>������g��_>�-�;I^H<��׽o|�<�n�<b���/�޽=����	>h~�<W�3��a��O��N�`���q(=ö>�Վ�������=5��<�?��$��:���=�W��tQ�+5F<E�)4����p�5���]0�8���� ;(�=�[��?�ܼ"�J��8�<�f�</
�_�4���)=��@<��=�0%�.W<2#z:4
�/jX=�M@��
��u��0�<O3!�!<C=>�K=.�!�l�+�L�X<��<GX��W������ں
��`2=��3�b��:�gb�˼�=��N�iE=�ؼ�,���v�5xG���<�+��}�=��<����CT�~�<�u=NϪ<���x�S=B#�o��C�=���Ƽ�]e<�u�����<l��V-/����<R��� ��=�
=��=+�$�2��\�o���a���*�Z=�=^}�(�t=lS=Ɗ��N�<�:�<�+�;M��<��e�G�8��޼��|<O.�����Hj��5༲��<�=E�<lb�<<���͜<唺�b�d<,�fGv�h�=j�=���<L���C�<)�=i �]ת��&<V��i6�<b�����<8��@a�<�n*��¹�1@�<�@=a1���k>8��SR>*6o��m�=��𽵹�<C��<��1<h�=���$W�<�{��.F�= w�����;?�P�a��<��k=��"=��)�h�>�m�=��B�v��k�Խ��"�]jۻ��<p_��,C<���<���/��W3�=5�=!)/=7F=��m;EW0���!�݆D���`���w<��6;�+���U�<Pr~;�~)=�\<��<�2��R�U����<���;��a=�j�h�=ܥ�<��F=�ԉ;M��<z���''>=���D�o�#$�<���=�J����<F�x=H�;�T=ȅ�f���r���=9��=��<�.=Ȣ�6�=#rἘ��=��[��A�:��9=��v=��=�s�=4NT�K�9R��'�<M������.R���>�<2=U��</�սi�<X�Ѽ�Pf=�Q��t�,<�]< jr�l�)���Ӽ��"=5��;��4=��4<W���x��<+�@��:Н< S$��Q�<ri=�B�|e`�E�0=���<:8!=@?�<�	���;0v�<�|�<��g�ƿ�0c�9`A�<�I����)=8���ge�<e=��<;2�<�ڄ�>v����<�q=plϽ��<��I2��F[=o�@<�����[��#�=��_=��<X�ѽJ�E��p��߳�=�b>�":>�x�v��=�|)=P�.=�*=�(��0%ݻNSǽ�d=��C��~�=�c���e��9]�h��<�f����g7�Ul�>�'�=a̠��9�uDj<���<\�=�v=[2�<�#.���>�%=��n���!_ҽ{e=w��;j�˻����x5�:�<=t�E�h<���e��M;�e�u=��<ޗ��e� ��ڼ)�=�=�����,�<*0��ؠ>�=�L=����x������q
�=\� ��Ƌ<�r��G�=�`u=)B�=�6�S��]7��Z�=��㽱 ��|��=���=KoX<��,;\���=��ʽ��z={I��@	���=V�>S>���:	���/=aȽ��==�6���"����<���=#iQ�i���b�����N��`W�=
�<��2�!˙<��=��=0��<������G�RS��4m1<�㑽��3��3���<�<��I t;Ԍ�L�$�u���7�Z�r�;��H=�[����<<_-=ґA<���[�h<�� ��<W�<����Y�s=�\�=)6�=R@�<�������o��^�c��wμՋ��E4/=	��=�j=T� �4Q�ᕖ;1LU���\П<nC���M>w���L�y<�j��k3�={�=���{s׽ ��Y+=&�=�� ��F�������=����ަ=�?ּhm�<z��W9V>�W=��~�B7ֻ��*=�|�=��6<���T� �1_ٽv�O>Y�>VГ=�凾��<>o���_>�5��.�ѽ7kP<.�=J��<�%=�'p��ɨ:󛉽��i=@�P����;�='2=���<>W�=���;��";I��A�=b�:�h�<F1 <��I�=T^�=�AL=JA��4���(�=����.��2�=K�=���=Og->�$���L�VϢ<���}<˽��L=+t<m���=F���Ӗ�;�o���B=�y˻���<�;ɾ��1u!�Pf�<xl=���<�A<�5�<���ρ�#��=���;>�ӛ5=F4[<0�=M�	���=Q���B�<�D=�B�B��<���=*�z= 햽υ����=<�c;0�g�Re�<�Ǟ�}�<��2=_� .$<}MQ��=~t����!�?ݼ��5�A���՚;ꈃ;�?���r
���;7,�<�6�<����l�3=��q>O�=Į��`���=�W�<��l��4=�N����<_��=lP�=a�����ݽR~k=���<1x8�=o#:��=R�=�M>�ܦ=��e��'�����<t5׽�;T��r�<-\�;��;Y\ =�<\dO=��J��/�=���Ku�<�㢼�8�=x�=��>��8=R����\�ڊG�H5��s�0�ع���P>�J�=� �K�;B�>�H���p>ҿ��'(��/�=j=6ݷ��4�=CI<)����u���	�=ӫ[�1��;0�\=/�6��8~��T�=����r�<��J��f3;O�=!���h�w=E�p�S,*�T!J=�j	=��]=����鋼��	>�Iʽg�������Ci=��=}�S����<�=��=�� �f�ǻn�e=�u��A���� =cB�<� G=��˽���;��U=�@e��rl����{����)!=���<¶�=vNｊ������=m��Q��<�鼶Y������<�M�=䁕�pr'=�).;Sh;� }<�G��G>k��*='�-=��j�I0����>����=e��j�<�Hg�����,~����;�s�q;��w�qɢ�x�]��!:�f:���<���<S}�.$�<=t�~�3�@�<�=]�K<�TK�Q����f��jʷ<~�O��P�Q}��.>	���E��=��u��y<���=5�<�����ަ�A�H<�9<�^�k��=�ę�H��=	�=0�V=�]��K���<
8�ܼ�<��UT��C�G���<��)�h��=�Bɽ]��<� =�"G��Lr=6_���_�=��=�@8=��޽�~���s;��<�̾<�M�-����l�=��[<X&���O*>���=%�=�9K�Sh�;� <�ˀ�?�,���;λ����=n$��,�=�y�=\n���{ <Ü����+S�j�E?�=�潬G	�Ph�=A|��H.��Ti��K�Ի�P=��y��=r25�tW>!}=8r���v��e�ӽJ�=��=�5�xU��BH�P=�x?�N+=+2='%�v$���Ɂ����<�=3��o���=pL��,�U��V���3#j=��ϼ}�=V^��>.�=�%�A5���f=���s��`/�,�����=黏���u���=�	�A������� �z�=�LK�O�=�+�� ��=����;�2�ʻ`jA�2��aLd=Ӝ#��^��
̨;�a�;E��� o�:�X<������=���<�L�;L?[<�-�<c-=_��EI=�������)0e=��6=�}�<��7������=.o����[<�}X�|�D���&q��W�e�%>�p��N����A=�d����w��;�{�5my==_���:�%��;?t=�}�<�Q�;���9l��dнa��<��	�C��=�ǽr�<�g�=)@���/�lǼ� ���Ǽ�}k<��=刾���������*�Φ]��G��o��=<:=�W6��	�C���pǻ�X�;��	=<#<��ҽ��<�2n���}����%>��g ��f��=��½i�:<���ѕ����K����n�~���4.=����b�H�?������#
���;A�y<��=˼�S��5�=|�<��ݼ�	u�\�.��$���8�<O��<u�=�ɣ�a�=�=��?��y ���=Kf=�{Z�f!|�q��/E�=�y�`�=&�<S,ؽ�;�<Q��<�9��WU��~��=��<� ּQ[��(���.�׽pov� �x���Q�=O/����]=(V�;$8��Ϫ�������<�c��~Ӽѥ=���H�����4="��<@��FֻqO�;���<)�����=*��<4�'=?�@<:��<g����a��)�;}�˼�<��L}&������x7=`"�$��e2=:�p�� !��ݥ='-s==�@W =�<<�QO���+�=Fl�<�K<4n���9�<���<G�=��[��	˽�&¾��N�;�Ӡ>����,8��ӻ="��,�+�q�q��E�9��<6����������ܷ�=�<;���<�r��d��; h������"�;B�w=?d�P�%���=T����ܺ�>���� =�������0r=:-==:U��=�1��Z �-��s��;$��*���>���:����$?=���@T=�p���9E�F#u��;��Я}<
<T���L=&�<vQ�<�8=����ID	��)｢��=�'�<�C=����Y�:��k<av3�Q��N�ҽ+�ｋ�S=غM=��8z�սH����%�=g�ۼA�e<3_����;r,����H=��h*=��+�E�����Aj|�6���Ȋ���R=�M;8?�Vs=���*��iO���	���Ӽ1~(<�m��-�=���A Y=�+\<|�b�h��4V>��9�<`����U�<Y�=W�=-�Ϻ��`=�]�<�t�<�d
=;ue<���(�=�pP:�d0�<Z���~��<�d����=�b��b>�<��޼aN�<��<�Q/=2�*�y�#�ûtN�����s
��
]=@��=Уl<�O���Ue=�n<��D��~���$;2�q�~�=s,e=��@=U�+�G��=�+^<6��=d;���D������v`>��q=�$1>\F/�ٕ	>o�=?�Z=�]<a�<&��;�ͫ<j�"<�__�\Y�<�a�<����QTμDt�=��=��:���qj�=��=��q<獽##�p��<1�e��[~��ė� !U=X'=�vI<��`_o=��@���=3m��Ā�:Cu��%4<<�ӻw]ӹ��<b���&)�<-�=i�/<I�������+^=�"�=��'��ż2
�<��=��:?�=�Qh9GrJ<O���~j<��Ѽv�=*��������ļ���ޟ���*�<�l�<�s=*�}=^~�<üM�5�?�!=���=UF�೻�D1<Ƀ<w폽)M�<v��@L��A��:"�+>W��a�=,޳��f!=�����I=���O��3R�<Bw>���S��=(�ݽ�j=�-�ltb=.pp�q�Ľ#�k=�=����ɛ�����<誨=+a=4��<6�	=��{4#�k�ռ
4�<�-0=��<�&���=<(R\;�6F;�4�E8鼼l�<;�꼟��)�����<���u<ZS�����c�<+��=����A��<��=Ή=;Dd��G.���*���U<l;'���?�P���8K�=\�c=��ün�1�h�<Ol�=�->=����m����3�=Ga+>�% >%��$�=�ږ<�4=K�=�H;�v<Pf��&�=Z��_3=?᣽����b�@1˼�Ƙ�a	���1�<�j8>��t=GV��s���D!�<cy�98&=��w=�0��u#���L>�z= $>ZO��;�޾�M=U���bJ`�u�t�<��=�̼��`=89�<~.�<y��A��=��Ѽ�>D<�s�S;�a P����=�7���Z�0;z<��=��6�@��<�Zʼ]Eh��2��>7T=O������u��Չ=�~>���=sk5��(�Ӝ�� �<��н}����<���=n�o[W�Ɔ��>�;���'C�<�-��=?�O�@=�>�?ƽ���=h6�O�:=�<e�W=.�b��z��<��$=�Co��!;;8����<�Oؽ4�<往<4�=E��KF>~��<6��<�׼�%^�/8���c�=y���c��m�<���r�<�q �3�����)�qB=$����;�_ڼ�=�����<�=U�Ћ=����+�=�ё<N��� =If�)-�=��>�#�=�	�=½�CA�?=P׽�� <m���A�=S�5=x�=e�ڻre��h��M䬻;ZL���V�L��" �=?5=dO�=����ͻ~��=�׺�~�d���e��<'c�=�6��&={������=�g��݂�<R.к��:�Sz���e	>�]4=)��<;�m����=Ҏ�=���m.�֑{��޶���>��?>͏D>;��ޟ�=���>r[���)��n#����=�J���-=�a��ۘc;l`;y�]=@N��P�y;��ۼ�S��T>=���=��<T����ꩽq3y=���9�P=�滌VI�?}\=Q߶=��=����5����-=<q�<o�,�(Hļy��=�[j=!d�=��=����~���%g��'=i4���==�p�<^2��S;�!]�l�@<X*��NQ=�u@��i�:��=�zL����;��<� ]=����M�!;�\=�5�p라j?=�g�=A3@�a��;�/��6��<=FW�ӡ�=��Ҽ7'Լ�:}"���=!=�=9=nR�L��X�g=�d�<[UE���+�n�;��+�;ӂ=G�<k�;n`��ņ;:��A�A���I=�"�<M�=�/=W�=�>k<��� ��;LK"=�*.�^�=���C,=�O>��=4���$�5����<�C�=�OS���L=����Ib=Z�x=��H=%F{�Y-�9|�=�i<
�,����*)3=��m=*oo=�d{=��i��*�=��<2?z����囕=M�<�2���t�;�yK����;�u
��z=�j��7��6��O>�=jp=[J�=c`�=��	���8�](Y=��H�a�~�ټ�D�=�`�=$>=@�=ں�=P���<)�V�e����e����>=�,�;r�=��5=<D�� դ��_�=�e��W�n1=�Ǐ��b<?z�=k{<;I����ǽ�����=�&`<�A,=�N�t�����5=��-�٤�~!��Ħ��2�=���q�$r!<�)<\¼P�<L��;/!B�hҚ=)��ؼ�;�6齔��K0;�9h�d��<�E^�0��=��p=o�<jwt;�"Ľz�W��Nz=/R�<�ڪ= ���J��S�=W|��,:=�B�\|��j�:�E�.;�<Ӳֻ/��<�~=^;�<ɚ�<����Tüh��</Ŀ���Y����� :etd=��=kv�F�S������h����<a{i�t�����μ�b�;v�[�
~&�4����7=�]�<P_<zj=5��d�;�7����<�29�	�=2�<=�켥�����Ǽ�	%�#�0��F>2Ƚ7#Y=�o�D�=�H=�~�¤�Ia���r<1�=��y���k=<���ǒ=��=��=[-\��q���?�pt[=*;��q=�����2c�˳�<�~�=27����=��W=��F�^�5=�ŏ�ݍ�=5E=x�=$:���ϐ�3,�<�}���!=y���Y�;�<N;���o�<r>��D=C=��+�j���=�d���8�<]�=;�&����<��+��a�=[<e�;��<�(�h_���ώ'�ϼ�)>{���-<v8�=��A=^���J�C�����<O��7�=r(A�M��=�]�<k��X"�(?a�F�<l��;w=<�̼8�$�Ì�<v9�;l�'=;l<7��3m����֢��E�=�7�{�<��;=�w��~g��cv�X�p��E=�t�;�>C��h��=v�=t���J�# ��.���0Z�vQ:���!=�����5=�r�=)G^�5�^<ˠŽ����H�<5{<�Ps=t�����=q�;=P�̽G�y=��;����=#O�<��t�W  =�g�!���U�(<��=0�%�b�=�4�<5�<��>�8������%�����3�j���s��<�4b=>+;�d������2�=��;e��Tڏ�4�������B������3�=�C齴ѡ���3�^��ŮT<�ѧ��>j��W�<�_��/���鞼5�=�̈=�@����==��H�ɽ6\������=�=�̤�U<<}�=5���߻Oۑ��	���v���<J‹�s�����:sk :@a������d��+\,=�\<�ۼ��K��o��p5=�j'<劯;M�<q�2��I��lս'��J>I���a�&��<��ȽttݼJ���������:�@��PP����G<{� =N֝=^��2���|�U��P���s��W%=:�=Qk<�2�=��=�P��޼|��ħ����<8g'=sr<]H6=p�9=DG�Ʒȼ��������y3n�=V5�J��6[�=q라':�=7�ƾ�Q�R�oX��f���I��<k���%���ǝ=���<�M��2�Žp$;��=��-��l�j<ʩ<I��=�@{=�i =i��ʽ̲��t���ޘ��i?=�6=�&6����=��=��,�v�Q�3(=��=i�>=P4�;L�%=���<���<�2p��ܕ<�z�;%3�<a����X���;1�;�\?=z���c�y�O����o��|����Z=)U�<C;C=}��c��=��?=�.��\Tƽ��d�<XI�Q��;�'~�3`����u=��I=��=)9Q��Η<�eŽ�l�Ø����>�<��)]������
��`�˱��vs<��=*��ڃ<���&b�=�%�=+�L=����~}G:���<����u�&��=K�ͽ<��<=�2<��O�缷 �����<ݛM<�E;g�)=��I����\V[=R�;�_����2���!;���h�(:�M)<܈�=������;��f�kC��$��Vgv�+�`=S�<ǃ�<�:H=em;=X�;`�b�j����t��YS�����=k�<�Fw=��7��<&D�<��=����ec�Tq��	J=b.���=?�C��B6<�� =5�ٺW�g 7��.�d�<�Ԓ<0��;���=V�g=�?����p�V�b���'�`}�;�c=��=��#=��=��~:f,P�������荽��
���=�=�D�U��=�O�v�< �~<�x>�I�ýpƄ����=O��=Q$%=yC-��3��qˌ=�Pq�s�)�D�=����綝<�ʸ�y㻻��<ۻ��8���<�tݼ��	��Q	=V�#=?����l�;�U!������<�fj<���9�#l�����=ݷ�=Dq!=)�y�h	�<��=e�<k�Ġb���x��vv�<��'<2DA=Z��<�D�=
C=������Kн�L>�J���n>n���>5���=�8��~��C�;RvĽ�aN=��X���ü�y=�=��7<0/���3�=�\�<���<a�l����=�c�<4w����Ψ��~�<v�������y��=��S��<޽�y�<�=�����p��Ӝ��i{�=9�����Q=$�=n�s=�|"=�,��Vq�( ��C%5=+q=��d=6)u�V��;���&��ь=���d�K=�A���L=r�����c=��b�q�t;v��ء�=������*�½��=�Q�=_4�=���<��1�q*Y��E�<��3T�<ӡ�ԏ<z];&"=�:��R������ q�=ʽ<ț=Ý�����J�Ѽ��L=�A�<�^;)�l�Y�=P�~��'�<����2�Ҽ��;���=�d=���w=���<���Xb]�𝇼A	=摬�d����,<	<���<8	[<��⼻��<z�:���쏻I* =?*�;i>��tq;��=��}J<_��<��<t���.�<��}��7��P����="&e�t�<C5�<��<I|�:8ǽ8[
=/u�K�)=�T����PiԽq�P=	L�=$E�<���K+�=",1=L��=g�d�JO+�.�ҽ�<A/>��>�r0���<��}�"^�D�6=+�
�D2�<,A���ey�0��=>�#=U=E�$=�.�#ߌ���C�Bj�^:X=`�!><�?=����t�T�=��=�_�=w�<�_�ꋮ�]�h>1<A\}>����h���j�<˰C�"���u��F\=n| =؜k=NE���k������V�51>�#��"��<8}i�*�~�5��<.��*ث��=G�T����=i�;��U�=("��
�)���h��k�=cd��=���`x�=|� >�n>�M��V�g�U|��H�=��X�0m���B���|\=tOȽ��M=-@��,�1=0�L�36=cdL9�ڏ=O�����=��ѽ��a=�⫽�Q���n=@Խ<��E?�=Y;���x	=��ʽ�e�<8e<��{<�b_��<~�<_n=K2���=,��<�)=+3�;^wH�Z���m�=��нj�м0�H<�3��p-2=Ap���ؼ3�;Hr�;�軴&�<����L���_F�\�?�q<zw<�t�W�<ߑ<�#�<ɻ<�NCi��J=��>��='� =d1��户�,u>��㽤Ӑ:_���TL�==r����<L�7�J�<�#Q<E~Y=#�;_��rv<�fW=s?=���=��xA�;W�=�������?�g����S�=���%�<�.���M;��`�Nh��ݑ�<�I=H�����<Kp��3\=kk�>�=���=H!��36���������#>�`�=��>�I�讔�n}ག0=q�h��l*����r�=#���ޜ=��:�T��<�<FLb=�EH�ym�<�G����U=��!�A��=���; Cڽm���uM=E�ѻS�=F =,?�����=pb�=��E<t�3����/R�=a���Ӽ���hV<=yВ=�8�܀�=����2��ƽP��=�]�L����ì=%��~(�;��!��ts��X��m==����=�K.������f��?]=�]<�͓�� E<��Z=�é����=\�����5=�=ҽ�u�=4y�j�0�D=���{F��@�=�D=���IGs=|6=x=���	f�I9�=S2�<�7=ȟ~��)�����;�����=/�2<��i���P=t�]��!=P4=,�!�=��}�*a����.�|��<�M��HA=���<Y=�[�4v >J�3�T��;O���[��j.ѻD*�=�2b=��x<��ټ�v�<�}=�k="g콝�����=w&����
=A*���=��7=f�p=�_>j����� ��<Ul<V�ٻ4�j��g)=d��G��L�����������_�<����=�#8�ky�<z��<J;M=�"�=�Ž�栽b4��*�<R?��Ȁ�䙳��B>Q��=a^=���=������-�K��y��.ؼ��?=��)�Cn<�ڻ��ǽE��Xk<�� ��	�=�:�=VX7���5=돶<�Q��z6?��G������;L��<�d!=�����y�tw������諒뫕��ҿ��	>lfy�+�ݽ�՘��u�<u�l�@=�k�=�VJ���U=q/b�"R[����<N7�^�ȼG�<��E���F<�	�<��<?B�;�F�=���;��1�D���4U��{t<�߫�ܘV����:*W�=|ִ��BV=G+����F�"ک;��p�ܬ�<�:c=,��=�~
=ږ�=yr�=�!�Y�ǼNO��ˀ�}J��ٻ	$�����=U�Ҽ�*�<Ur$��=+=�ؓ�7Bs;��<���s4��=<�G��s�<���z¸�Z�
<���;�w>��t�;<�<1lt;�ݯ<�>���=xC�=Ь3�7��L�������t�1Q>Ֆ���?p=@���S-M=��E��+��j���K���<�X�=��=j$�<6��<|1�=U|�;���< �1�����=2{����=؂L=�NW��n'��c�<�8�c?o=�4h=q�;���v���1d4;����÷=�#��A�<�O����ƽ���<`N���;v!X�s@�����Ȓ{�o�L�j��=�n= ٠=�v�KȽ �a<�{;�|K�rp4=蠌��l���/����W:x�H=��m=|��=�Ҫ�K���/m=�� �}-<�j<�'=��j=G��A.�����:ޘ��Y���<73�;`5�(+�=�>�yF��<���^_���N:��H�I�~;Z9�;T���}F�����PY<�"<��$�`���pn�?�L=�:<Hrƻ� �=7����6����.��c]��� �H�����=�mX�:}�=M)�=j�&�|(�;Yz��PԽI������o <  =�� =Jˁ=O���A���\6���ͽ_tԽP%`����P�q>�\�=��ɽ�@����j�������a=����S���]<GQ�9 c��7=g]q�֝8�
y=�Y��==�z+��5�<��:����	���ĽY'=&��r�C�E��J��j&�6��=9��=��"��,���4�;1L�<}1Ͻ�mս�j�=��̨�g�{=ۙĽ�g�= �<�{:�ӎ������轣��=��\��c�=A�B�/�==`Vd�Բ����d�Խ�K=\�<C= ��=Ǹ����;Q�+<���!;3�IPͼ�N��[=`�b�Ɵ$�P3:�τ��͍��K=:_q=�&=��*��|:����:Z'];f�r<��=b�'�W�N����Q2�=Z?���=�,n=�=��Rp�1�r;d$���^Ƚ��u>�B��m�<��:�:;=��'��ͻ>���8��|�h�'�@��=��
=��{=��a=)ᔼbT��n4��[+���m<T�[�Ds<�IF���<`Ź�s�H����q�������+,��qw;�=�=~�}<�=t+ ��
��{�Q��q{;<�%ļ��=��;��<�JF=RAk<����#ϽԶ��O❼�8��7%J;���=q�d=Qtt=Z�<8��;Э��,�!�[=�0 ��
/=q�0�$+=��;�T���<�;Q��<��!�̝�<�;�0
�֩�;�1�L�1=�=���A��O ��2<��<?��<��=����j�<Sd+�ϵ�<�����c��.Zd�/S1<�!;j�=&�#��x�<��=�k��k7�����-#��#/��n7<��=���h�=�R�<8v�W��c��<g�7�N?���F�>#ʞ��
�g#=C��$S<)4��X��|��� �����"��s�>y%�<g^=�$��A��`�=Ԇ<�T��f��=']�U�����+<��ż[G���0</k�EQj�7�⼰��;J���PX0��w=��R<;���Ƚ�࠽�Y <<'P=c�=��=�];=a=�Q5�3�7�n��ɰ�[6~=�/~��u%=��=��<ҥ=�F�=�|�.ֽ~e����=�Rټ$�=�h��O��<Zl�����<�e��ʼ�"v�GO�=�=�;�s==6���ɗ�2�U<mD���۽K���;v���>+1I��S�<�d�<�
�(�`�h�:�����l�g
.��t>&Ҡ<��<�"�:�>=�^4���L;Ko
��E���p�����=��=0�u=�c=�����(�=�^½���yq꼏İ=k�=��=s�W<��ɼ\��<lӼŕ:HY�<n2j<ˡ����<�t��$�<�6�\��<剳�iSC<�P<��ȼAŲ�LcX���	=s]c<��ټw��=pf�����h��t�=l��J�<a�Ƚ=��=��=-�o=,~��:.���,U��c�97i%��2u=u�I;����b�=�m[=Ɏ��춽Z<.U�=� ���9>-_V��fN>"�2�����r
���v���=�z;o=��<�����r����=Hh���Hü��pT�==�:���;i|�6>�<)L�<����1��Bɼ��=��ϼg�X���fI<$)M�9Z��(ּ+Ԗ�����>��=���<K�<
M4=�ۮ;8�b�e<�0	��l齦�~�
=��I<`�c=�q]���}=	i��B�>��=(�#���$<�1�=�a��xh���.=�a��#Dx�U6�='��<��=��1�h�=�v$�fa�=�E�<�	޽Z�ҽ���=Gْ̽P=�[�</@����=���h=�M�<͡���X<��䮚<��=���5(�<����F�:�.�<}|�;u=OҶ���i=(�C<�>E�=�ǒ�]A<Á#�Y�<ʽ==
��+=ɾ�<��J<�wK��5<�y��=�#�P��(��;��^�(�&�b*Y<�{ļwjؼR˯<1�T����<(U=Qހ����!�1=-5�N�< �:�k�<��< ���WVK���f=�S����w<��==��<ї�f�˽�y�=�&:]/�<7W8������ c�3�V=j��<zF�=ޥ^���r=��v�fM�<'�^����Kg�������;=�;d=̱�	��<���=QX��4�=FŽ��=�v�<�Z=19=����k��<%:=�;��=�^�=ļx����;���<�E=�>b�NA<�-7=�.��5��=.�<�%B=��׾��>�!�&�y>���=�}�T[ӼԻ�b����==�g9+�=����Q=��.<;���X�=nk����<܀�����;h���2u</�ǹ\�=$kսT��=]��<�𑽸������ѷ�_|�;vW�=�ƒ<�'>h>g�Y=P�<:��=ŔU�F�ּ��Y�ψT��v��>�;H�߽(@�<�d����{��F���|�R���(�=�ۨ����=�
�C�<)��h�;-�<�ʗ����<M��=)Ľd�<��D��i= �_=`>z��:�<�9�� w==F?=�k]���=O�׼9��=֮��7<�<)��a?=PU���<¼/��<̪
�;���3���I�D����ؼ�p�<��<�WL<���<��=��0=�n(��{�<��������Ǽ���G����Y�5t�=�l;u� �2E��z"��>uܩ�<�� D&�u�=nG¼�)	=a���+�<憛�d ==q<��u<UpŽ�%��$=Ll�=O>i� J�=2�h=�F<󂣽!QE�AA��
�,=݄ν��=iw��7�;�-�^k���1�<��=�����<\�<H�e= �;�)"�=}q�<����o(�Bu���,�C7�=�	>ţ�=�Fһ���=�������x<+:��A<�d�ʀ�=}k�
n�=���>=?�	=�֐���<�K]=�����:�����=u	1�˰����A�O<���r#�=W�=V�.��<솏��Ҳ<�>½�Y��<6={΄=i�^=��; 	=��>�W������Mo�*#�㕽g.�= �<�/ؽ�ۻu,N��L�=���<�QŽ���wi=������=>�<褽�s�=�Jr=
7.�ڃ��\<;Cv�;�/<�O0��:=O�9���YV�o�4��zX<�D�<�D�<isK=�#;�f���uʺߜ˻�3�=Z�m�'�����7P�=�6�=�"�<��R���[����;G�)��pb=�T���6K=eP�HU��oI�<p�}�ռO1V:\+=�׼��;�V�<c��B��<ĝ<��<�]�=��FD&=8���@��^½��8>��=�UҼı�<ݑ�<%~��2��=��Ľ���a�=���S�;�>��w'=-�=��Z=�����
��y��r�<s=�<����L"��ý���<�<Ї����8�E�2<�=��!=O�`��=T�3;�q
=�����C�T�`�6�;*�/=�2@=� =L�νb+;>�T�����$s�=�,��X�� ��l�=#�ƽ�yƼ�9i���8S��g48�@]��(~<O�=�jv=�E�=�����Yr=���J�=�d����(�Ǆ;!Z�<��<'�7=qd�pQ=��r�2�$��W�w���2�.O�=�!����o�RX���=�'�;�=�(�<���UoK=���`<8]�<d����1<i�c��7':Mk��w<�<L�@� Q=z�x=+Q�<=���ܼ��`^��P%A����o�<Z�R���=������&�׼�=���Լз;Pc��fs=���O�<���=������f=ƽ���_����қ�<���;��=���<m����oּ;��<$3���=�ּ�H<|��<Yǳ��>f�~�*=�=�켎&�*ɻ<���<4��i+�/��<*[�컗=V�=��=x�"��Ž��ֽ�c���`�Hi<��R�f=cc�����<�Y˽��߼�7��i�;�\<'��<�ˍ=�]=��L<߰�=\��rb�q���e'U�>���P�y�V<[�=�o���@<�ݿ�K7�?�i�~�'=t�g=�b^�<BXf�����?~<����$��<�>ɼ�̢����=}Պ<=`=8c��#rۼ	�;�ޠ=r��cn0>�E/=����h��'�<4�<lp�����쫼��<ȴJ��<����w5<��=�j�=����-<�2� d<�H��@�=�ù��\�=(����C�;jp=g觽,�g���;��B<������=�y�=m��P�M<�P,���y��;
�1=�����ҙ�.��<&<켵�����=4���ؤ�01����w��u��,x�;�y��6�r<�����D���<3�ܼ�/�-���	<���B>�w =p�ݽ�18=���}�W����n��o���$�=m�=���=��R��Oo=���;A�����Ž9�������A>B�q<t���8�M�vV�Kyؼ��=�81���< �<�J������.=L�ɼr-Լj,=�*��#��;q�4��MY;���<�P#��M��@�5�=�G���ǽb�<g��<t�߽U�>�~�=�3�<��=�m�����= ���6�#����-=���<�s�=�"
��|<��<�9�<��=�1ӽ4 �=g����=*
�=�}�<���l�G��B

�2��T�={3Q<z�=��������H�D(�<��a��Y��#<���d=��式.=�KG�U�޽o���͛<�K�:_�q=���<g����<���=X`�<�"�=�2#�Lط�����s���ԽF�x=��4��y�=ݾ۽=�=ܯ�<;q�j��vw=��<�p�k=O�=���=3��<T*`=mc�����Od}=�u��V�w=�*���8=�C8=�m	<�缚�<~��<?y��2=sA=��`�z6�'
0=��<�μ�#�q*�X�m�_�4�Ȼg}=yF=j�o<��=�u�<��>�V�#����=��M��!$=\�<-3�;£=��9�)��ZI�W埽���<� \<+�<B�;�ˋ<ە�=�>�=�Z=� �l)��J�D=D�-�t��=.���]�)�ɤv=P�T<��<t=7��[���	\<�I=�"=���<���<px�;H����i�;��<bO<b��;f����+/�MX`;N)��d�<4�=�=y:ƽyZ��L7=;���P��=�y�<#Y(<+���n"-=�.d=@Ƚ���K�A=LpV��4=Z�4=A����Q2=|V�nK���3x>���,{=�h'>f���>�}�=uo��|#���=������M9ٺ��g�m�:q� =���;@��=���x<9�=�*p�,�H��h�<�ൽ��W�+<�@��W<����
�����������|=�2=��@���<�wԻ5Y,<��8<����U���:A��<D��=��`=���<T��<rJ�=���Y���g�����<�=Sb=ń	=�cB���ּ�!�=�����м�DL��BN=�Y<d�=��4�E��=�$��B ;ɇM�8A��6��ڛ�<ق<QY�<@��<Զd=�<���=�˽�x����B�'��=���<��w=�X�;ռ<HG�&64=����]�'v���:�=X(�|ue=�LZ��Pۼ3�6�I�$<2H�~�ف��v��=���ۼ�_`�	�P=�ֲ��M�=���ߎ��BZ��f��=�}Q=g��=xL�����4?=���<�4��b��99����2<,�$=�<Z$c<bm@<E�ʼ�d�;����;#(=�m?<��:��n<�7�YV��VS��v��<��!�~w���a�p�=�_=/]�<_����
=��=9�=Ǿ^�����g����<�ו<ݜ�;�~�<1Ї�E��<Iǽ=*U>|F����؋�=���J�>����w�Յ>��1<�-s�?��������=1=�\g;UU�<�X��'�=Z�ͽh�^=�p�+�j�X����K,=�"d�1�̼���w��=t��y�r��	��q 7<% �<�`2;�v�V�=���<��T=�ٽm]����c��=X*�gx=�Y<B��<��:B{/=����ƚ��-<���=>�z=��n;\��p��=T�<j1%��9�<,�G�v�u=��V�Գ
=2D��Fk��xW=3��<�Ou���=y ��<s=K2Y�B��ۀ���=n�=�Y�����e<�\�Y���	�=V�8=x���ս���=g�F<��輍��<�̰��II<b;=�E��_[�g%���=n�\:�}�<�T�"��%��<�w��߄=���Dg"�j>=�^
<�R7��I;�`����)=���;���<B���Zx�C$=pJJ��]<.�<���<�f߼R/Ϻ�c$�]��<�J��2�=�Ҽ�m	=?T<���9�;;=� ��sż�e����<�#=��<�Zh����<�μ?�==?���R�T�$.��/�~=%=�A�<է�=#�<����.����=��>=S��:�6{<�|b=�:�����=�$�J̡�H����I\=9���d<$}�=d�����!<�}����;�*��i`�<�3�<�YG<�I:=�-�}�.<o3�<b�H>�9Z����	3�8
�ʢ=�(N��k��- ��Ź<�m��mJi=���:Ǥ�=�V�,O>W�&�;��=GM>�b6��ˋ�톿���ļid�;[9�=U㻲徼�(���=Y/,������AP<5����g=eD��Շ����<�������h�=���N������#�=,^��z_�D�ȼhƼBT?=���=�,��ҙ��>V]���[�5e'>���#�|��#��6���� �ʞ���Bv���:�v3=�p����׻܈��=�e�=��9�9�%<?�l�꾇=<�(�����=��?����<1�>tӽ��b<PF�6,�-�=~r	=�Ʈ��oB����<��=,̓����<�m&��=V=D��L� <��4���=�ހ<�Ά;GGڼ}=���<��<zz/<YF<�����S<�^%<��;�q8��YD;�q��7(<\�}�{=Fdl<��<B �:c�)=mހ�iq���;��2�: �E�"k7:�w�ݻ�=�E��N�<.���}r�=O���W=#Y}�FG�����������;�.�<��z)���g�=���=�z��|�=���;�0V�u�m�{.U�Ց��ʫ=�dM���K=:�<�A�<6���6�����;�A:=3^
�P��T8>YH�<��g���=+΀<�I��Tq��~M��Q��2ҽ�x>�}���l<�o�>���AT�w��<O=Z0���;�Ņ�zI+=jB�;NͶ����%v�WV="�=0���U)���`��ד��ݚ=�b�#B׼��8=��=��%=}�=O ��mX?<���Pw�<���ꔼc�B�n�=�ˊ=X�=���d>���{Yj���T<y�E��'��{����o�;ȓ������:��� ��0@=�~��s-����<$�����+=e\,=�$��'�����<��Q=�Α��Q�X�)���L<T�����
<K*�G|H�oS=Š��#��g��Ih<uQ�=�KJ=/\�HN�<�ŝ��A�=}���+ݼ�[ͼ��<�=\��<8��<P��<���<��9�w��;US�6��+dٻؚ���O9�Y�<TqJ�fF����n���K�<� �9S������=Wmk<�J�<�wL���󸌏��Kt��eH���VH=��=njH�������<=����yD=ej��&Ľ��<�Jl<Z�<�o�0�=F�o�f��<�
��=�½gc���=<��#=�
 ��q�; 0:����<!�k<Ύ��}�<X��9�Z�;_^=���=�=�_ͼ�jz;뺖����b�
��� ���=B�=��>���JK@>K4���H&��M�=�ҽg6k�Ex���H =���������Ofw�4��;�e��&�L�eH}={i0=�D=�,Y=�E���>�:W��{�,=��C����;�N=5;����M�=|���G�;}��q��N������.=�~����<�r󼖲�<���=(�=4:��GC=�����r=���:)V=`�`=`�)���O������=u�$����<|�<�w�<9���5���彗����Lؽ�c�<����G<D�y=ju9=�7�=�<V
7�*��<h��;{%�<Z)��?E�<�����L<ªѺ���=r�߽'e=B �/� ��c)��4�=96=�eX���=P�=�+�!I���G�y����b��e�<�b��~���Φu�(ĸ:��i<+|
�@���$�=���<��<8���v���:a��=O1�<�5>�W�lo)�S%����&VK=_׉�B���/_=a=��]��̐����pg���1=�/����C/X=&�<,�B<�
.:��F<�!=:Ƽn�׽r<���;�~�;�
:=����Ἔ ����l=�N�����0��;��E=����@μ�̲�]\�<�4:�)N�<�A��j�{�;���䏪��B?=	>`�:������#Ua��>�vK>.N��(���g����<�oZ=�e��d�7<�΢���k�{�R�ĿW=C�X����<��t=�56=z����=Ƭ���)�<���p�=�f\=������
�Q����~�=@\�<W��B��=�函�p��<�=9�J=��-�cD���0��c̼|�;&9�<	G�<M�6�� ����5=Ad���ή=���|=9]_���ӹ%��Ŧ=K=�و����L�����c=zʼ1���X��c�����=��S==ޔ�k�L<-ٽ��J��!����Q<Ԥǽ� �=ZO�=Tּ<��8���Os=�MQ�ϗ���!�^/����A�	>q,5=����8�6=���<��p<�b˼<H=6\�<G9=2տ����^�d;O�W�ֈ�<�"#�zrؼW�����~<���2t�8��q��<ɘ�G5�=�T��'HX�Ź=J�i<�@��܍=&X=�B��I��=�����T�=� �.�`��,�sw7=�	�<	�л����*;�j���4;�->�(�y=�?���%=�Չ� �*=�	�=/$3<� �%�<���d�����ٽk�=�ʚ� n)=�W����<���-=�ɫ�ۓ��L�Y�t=CJ�^��<�<=�;W�Vd��#輆[μ%O�;���=K��pi����<� T;q<I=烌�h�=�;�I%;)�ԟ(�x�����N��'���=p] =p��@۔���U=�;�=�-�<����h�&<��.=B F=�J��?�q��=�Ȼ�zn=~P� �Y���=���<2�~<���2{�$��<~'Y��r�<� �Ȣ�<ґ+��u<0 ��8�]���Ľ��=ۡY��
B���?�ɘ�!,���<�Sw�ބ��1�#�ʤ�<j{��6!=,Z��ɋ��
{;��X=�]�u���8����A=�s��(�{=�`��sH��v�4=Sp�=�K=	"�����&%=Rى�
6�<����(����;�����3�;u��<CL	=�B<�)=�U=L��<D�<a2F;r�=�b�<G를K.��u�<�Q =�I��A-���ܼ-8�vh�=!N=0Rz��g��_=�<��w=2��h���<D��<�2>����Ǻz��L�<�4�^k�=܈�=�U��햼��བྷ��5��>�={ <Ģ�j�˾U�>^z�=�7���L�<���%�K�c���0��=�(��ϼW�ǼR1<0�=8��<XJ��b>�:=B7ϻ�e=*��
/=w��<�`Z<([��{��\�8��\��2��;�*�<����5��1����~=�g�=����CU�����=?�B��:y���9s��$�=�8�<k���qӼr�^�xJ�<~)ȼc�<�x!����Eϼ�w�=,� �1�0�5�4���=T?�;�ُ;�Cs��!,=�o]<��Q<� ���>a�qb	=��"�z�d���;}=�e�=D�4�¦=�ؽ��ؠ�����=���~=4���_W<����"�=��#��66��}�Q(�=�~�U{<b�۽Ҥ׼�>����=�껽=vW�����>��W<bMz=�iȽ���<�ø:�T�=����m�ɻ�����=�J��-�=Rļ�����s����J�.<�1�<[p=��<y�Y�6��<�1w;���U��o��<u=�r =DUлS���wmN;�c�:_� ����<��һo"�=���o��)y7���V=*��<��= |<_=�CѼh4�;���ڋ�%�\��9@�f��=k�ܼ0s���ɚ����=3gi>�����r7=�2=�T3� @�>�n��O1�G�=���=z�X�n&�<�齻�=CL?��J=y�ν��=�)<��1�w��=�����gѼ�(޽�[Q<�U`��|&=٪_�Mv~=�=�o�鳁�*�@��z;IbJ�#���qu=���<m� =�������+&�[+�=����w=�l$�����
���M=��F�s;���<�[�=�V`�'�4=�K�L=�=�K+<�� <�fJ=\�=�D�<�ƒ��2R�e��;?��,��;��<t��y��=ꦇ=	kv=J��7^e�}߽���=M��<�󀽾���XEѼ=����=�:�=yA�<u(�<
��â;�����&������<�V�=�#=��U;�����Nl=K�<nP��s�<�]�<i�s;9~�=�-�}�|��z"���)��s�=q]|<E��RK����;
'�<��Ĺ�����=��;(�D�Q�3�<F��@sP��G<���R�<!�ܿ<��#=R	�<Զ��ъ�<�)7<�Y������K��n�y<
�k��Z>�=��=YG=�^"<��;A��Y"��g���O���O=�O�=��U=5"�<��!�<�-��/�==�7=���;����x�=` ˽�Y>�t��cܾq���P=��(��t����b>�꺽��f�'r����;�a��X:O;��=��f�$<#�Y�+��8�<C*z<��	>�+��5\�A��;&�:=�R=_�ɽ)i>D�D�ps���-���_=�ΐ=��=J���T�=��a����=�~�?��[�����`��L���P��=�-���/E=�G5�v�=��<�U��p�< �Z=7��<��O��;���*f׼E`�:�4=����W<�>=��i=~;���0�𵝼m=\�6=�k=�A��r*��Q!>r{��їF�u�=�.>��<ܮ;��G�<���Ej(=fz]���<l�<
>������rƽ8Y8=�t�=��Z��� \�!��<��=����k;[�;�o�;�f�=^�ýó����-���W=;]�=��h<!����k=<��Z<��='Bɽ��,����<=�=Տ��M��<{��<��<Z����]<*�ɻT��A����ټ��<VHS��@K���<<(�@��f<n~�<No;�L���o�(=&W�;oQ=�9\<�Լ� 꼸��;it���*ʽ;�=�a���=fR�񴥽��=W�}�Fc'<�$6� c�=u��;��\��[+�X�u<$P����H<�.�=+VǺ��[�V8���=7�>>ዾ��=�0=�!���)%=[.���0��|��)�+<��D=H�<;aӼ���<]ܼ�wn=š/=�̖�m}���x*>��=�(M�<>���<jt.�(g����]?P�j������=Cxa=?-����N>boP�����ȗ��$=k��SF�<�8=�*�<�4�<�v �覺\��&�F=�R�=��'��4��a�!�A�	�?�=pi��d�*�2<=w\��Z�*==�[��XJ��qoB�
�w=\E�<��	�Un	=O���!�=7>�nŻ7=�t������M.>�������=� ��[���@�~�iv�T�c�f�Y���r=�L鼷`}��.�=2����\<��=k½��ٽ���.�=)s9<xP����K<��<�=n���CZM��틼�uR�o�?=�[i�M�:=w{m=7�<X�: N���;������v�/A!=����
J���$%=�l�:��=w!=��)=��=�!��sE=o��<μ�����JX�<�s7;�S��C+��JP<r@=(¼�7ȼ0ʠ��.��%\@�|�z=���=.m���
���R<
I9�ͼ�ǐ��p����=H�`��᪽M7�<��o:l�<jO���g���yN=�]��z$<����P=~q=%՗:��J��f���m"��|���=�%"=�՗�Xcd�쬣�*�?:H�=G�:���w�;�=}L��Z�h5���?p=H����}�;W���5�<����0O��A`=-=c=�%>q"8��C�=�AM=OI��4w>,��0<VҾy����w��q��������<Q��=o��p駽��=�@�<�=F��=Z�Խ�/<�*��mԈ=F	��7��<��=�Ƚ�o0;QL�=s��d�H<��ǽ�3/��� �n<�v�=�Y4���|���k��T�;/nY<��<���@�5<���<���<U`=�lv=�X�=���޻7��;T�b=��C�ǡ�=IOd=ѡQ��$�<�l�<O���G��<�_��8J<�E�<C��N<V��([:=2�=FQ�{�ɼ̈�q«�{e����l����=�Q<a�7=��<䠽�Y�:�2�v�߻$C
��/=���<����&��� �<q>;���Ǽ2�;[�d�
�}�T��8�����<��H��]���,���=p��𜸼�⁼u\)��r.<�ء��/�<3�)=ch�=�}�=臔�':	���<,Z��(���j3��<݌v��8��q'��Wr=���*"=���<���N�O=��q=�B�M�:tv��v��;E�*���L��	i=V�L�mep=��=�P�K_�<�A���������Q߻ļQ=3B��S����Χ�Ȣ��F�z=��<RF<ZV=�ݹ���=C�&<]��<�}���)=2�$=�eo=����7>1���v��o>2=r�T=��=����󊼰!w��m�<`��t�=p��;x*�kC�L�2= �2<!<=��(�3�a�5 z�Ʈ]=�=�=�¯�s�;�#���
� �O�=�|=l�%=�B�<py���p�˞Ҽz.d��7-<o7 ���=�Լϖ�<rkP=�M��W�޼y�L�i�J�&=�Z��n�<n���cX	�w���<�j=��Q��#�Z�C��� �N6�=�8=Ả�|�?�~d=���f��fo=m���5u;?���SNQ�|ᐽ�f=�<�&ܽ �����&�#��H=<g#p=:K=;�!��<o���=a���D��An�;SSP<f��]��I�?��2
;8�����<���<�Y�1������=G��C`��!�<��
�J�N�� ����?�u%;oƜ�6���`�=�L=s�]<j�o=5����<�cڻ���Y=��;��#=�<p=��D��M%�}��1�;�B)�Ĕ��,�;dO�=ѿd��*��M� �N�T���=Ƌ=�命�	�;b�=�7�q��<�T��Yۇ������j�ɛ�<R����!�ʼ���<��мUt�;����
1���<LZN=}@���NT<f���k���of�<���;�.�<0i�=<��v����<�)���0=o����(�=*T��q�*XC�@ʙ<��=�	�&|�0.z=��I��=�w*�J�=��w�����E�<�1��"���V�<[�.��c*;B-��ԏ=�w.��8f=f;��B�����<;)�=�U�PA= u���	��	{<�̆=�z�����;t6=�D<���]�<�Υ���<"{���G�<����XϽ�������=0�V���<3�ؽ����0h���I�ץ����P;=m<�5�;̵���=�HC�S�=�߈�x�[=��	���h������V1=�d=w�<x���ϩ=0F���Z�=d���yz<�l`�����=>2
�,�<��<R^�<a�q��
=٢������"s�<|pټ���<2@�<��鼷�>�ev�<�y"<��[=�ή�z���4�*"�=�:���=່�?�\�P��?�<n������<r@E=9��p�����m=�܋�VA��K��:}��
A�=��>�R=�N�;�Ob��㡾�\��@<����S�ͥl��� ��4=5���'s�=�$'���^<�a���ýe�<.�g<sly��b>M=3����P=6�����)�������Qn$=�RN���V=����ݻ�� <}@p=Ά)��ۘ���F�2ak=G��s�&=jU�)��=���0�n=��Ϡ��d81���=��Ƚ"<i�����N=*�;G?�<G��ɺټC��<�K�=SW�2ð�%^���<�ℽ�r=)���)��΁(��m=�c���m;ځ{�&<&��2��y�c=�1<�P�<KR�;2f�=U(�	�^� 0Խ��= �¼9�	=*&e�ۯ�<��a��ت=PK��:��<��[�	R=?Kt9��=�n�q =I?c��m=Ԛi�߳b=�|�����=5t~�U5-="g�J<�gƼ��>I������'��=pvT��M�=�%��M3�~ڍ==�<��<��=tx!=,�!�(e���o<�^���;�ݸ<��<f��9ɳ�+���a��<Oћ<t�:��A�R�< ��:�"=��Q�<!]j��=U=/E��Y�=H"��X�=Z�u<�ڈ=g�p�&BZ<����WQN;͍H��+�=�]���'�@߫���	>�� �9[2�,.�|��}��;}8}>g�н��e��$>[��=\���@%�;z7�e�=�;��X��<=�����;r�M�� ��x9=W�:�`3,=n���I�K<ח5��'+�qr���ϊ=�/�=�C8�ٮr<���<K�=����<}�S=t]>=\ʧ=�i��Y�=���\l��R�E��<���;q}=�Y�_@��X�<�x��c��<��|�m9�7=<K����<k���1N�=�[������W<�u=�{'��-�<�������<&��|�0���Ҽ�������=�g=�����	=𒘽�E��=��=F�u�Y���a"=�J;9���==�.��y�Z<�錽2��<,,*�8�ƽEe��r����=�J�<sk� :��q�Ӽ���=5P`�nv��y�\=|��<���<��C=}%x�pz;�z���`9=�PE=6<&=�[�;���:��׌y;Կ=���kD�=x�a��fM�j����2n�گ�=Fkͻ���5�<_a�:\�R<p�1���=q��Y
�<1¼�.,��C=Ӂ)�	��M��e=ݲa�c�=� t���A��/	��m=�?�q�<2=̨=�av=���;+��V�;d�}�p��<G�����=����}L�=^�0�< 1��]h�����O߽F\3��]�>��<�K��M���8=��;=50,=zz�=g���Σ�3mݽ��T=���<��=B��%��3;>�԰�8ny�W��/�>��� ͽ���� ���֓�=�J�<')�= ��:>⵾�3�<
\v=���л�<�i0=[,���S<���<]���X�
�l��a�u=g�m�+�0����=��Q=�ع<���=�n	��������;�v`=�y���M�<,�>=��<'ԣ=F�4;�Ƚ�1=Ԫ<�<dΏ���F����;>�/�5�=��3��>
��1�S<\ʸ����<�'=��<ы�<�(d�s�d���3���T�<�۴�s�
=�5_=���<���<�z׼�,Z���e=r%n=@�
<��Ƚ0Q��Ke<� K=�V�=���^y�<����Z�(;{�=�������;�=&�i<��<o�f<�Gϼv�=a���<�D<��ż�������U��:��,����<���<"��8�G��`�ʕ������E=[<WR,<��=��5�A;4F�<�x����ƽ�=+=�������:�a��_Ὀ�>�總�$�<�!�5 b<S��=ѳ�<��a<|9�G.�@�]=ɶ���ⅻ�CF��>�xN>,�=Gޜ�X+k=u̼�+#�p��=�Q�PW����<O�<�a=e�=�{U����;rA���&�;�+�<����m��,B>��<��5�Fp>hL�;-C�׳�<%�O��L�������_P>=�D�O4>����6h�=ͳ���J��D��h=9a)=�^�<�c=��z������Ƽ��.=nMx���-<��ѽ�
���&=ɖ=s�G<��;;L�l=<e<,�<��1=7����x�
��<�k=���1��<?Z6=�W��3�=�J=�Y�=����	M=��6�=�ý���@��	����W��t9͠�9���=�Ϲ�LO;(}=�P��U�9���<�_��Hr��<�W[=~E�<W���|*�<��n=%v=���Ry�3S���<g^�<0w��A=G��=���@P�@|3=�~��F����=!=�V=if�*�l�H@�j&�� �W�n��<��ͼ*�<{}!�+׳��5=��<7�3����ۑ�<��X���6�Ab=����_���=�
�����?:�&�K=�e"=�����=K`ƽ�]A<�����E��彇�2=`�<��4��
1<�1<�v�p�;H}�<�8���<���<��-=��<�35��7⼚��J�˽�����+�=�Y��_�y��C����x�d{����f<w�����@��E	=��<)1��AN��@�#=*�q��#O�t+1��a'�7�)�,��=��=Ei�<��&���?<lʤ=/:��䧁>B5D�<f	<������L�<��ʽ�z9<���e�<J�<��n�=������=�1=�1��g�<Z�S����ce<�G�=��=h[ҽ�8<���=�>��z������\��*T=�g<X��lK8��%R�Ԗν�=�:!;��)=)���=n�������t=�p=��w='p
���<ߖ�<�:�&���\aj�[�=nN8�\ԫ<�g=�bP�7�,�,A=-����X�>T=	�t= ��T\�<��==3i�������ʻS�Ǽ�d����;�j�=G+���{��s�=w�����=L��;^Ɋ<'��<���<O���(���z��%��i�<&��;�	=��>�\)<�=`�=��M {�*��<N��,�����1��=��������1�W.$=c�)0=�x[�nA�=EYp<`�����\=\є;_�����/A,�ʕ�<�2����;�=�+#��-�:��=4��=��7Ժ~��=��t�<Q�J~y����<�A=Ь(=�=����1+=4}<�3�8O=�E��<�<>@�j9�;�=@o
��熽6�^=�&��z�/<�w�%�;%s�=� G���\=�'���v�<�R�� �d��=ȣ�<�����=�&����~=�؋=-�X�/V;�%�y�u�=�����%�Ѷc=���=���U���zT<��b��]�=N���N����"1=���=�&�E�9�8�������>��2='�ֽ�ò�]-���E�w*�;A3��W漖У�-J_<��ؼ^�W�T�<U�ź��_;���_�^�s9�����#�=�Ʉ=ZK��-����=Oj<��[��ܥ��,���-��=-:�=%���jv|��yý����peb�yC�?�k=�{���S�<͢�<�t�C��Cx�=��<�ҽ8R߽Qd���X�cU[=���=��㽏C����R�����Ϯɽ�㷽�OA=��2�� =�Tϼq4��'��*7�<t%�T<�<�z�<o�,=��~�2P�;
٧<��<�d�8�"<~�<��7�V+<�雽pH�Cn>���<`���i	���y}�{�!<N�H�����A=SN.�
�=Y,b=�᣽�������(�<z����޽\��;m���s��<�\�w���c�����T=��>ź1�;�G���j=�V�SR\=�k�,�z��1j��=���+?=+Ы�%$��pf;��A��
��1��uv���m���'#=M�<z`ټ��!��ې���,����=9��<_���/��=m�s��b��}=a?6=G�=���*�=�_�=F�:�&����<��=e>�����#Y=4P����=���<��=>�C�!�F�����}���i=�m��$$<"AϽ�(�=~���Lt�=�	<L_���AÝ= ਼�Y�=��+��90��1�<�^-=�=��%��8a\���|<\렽$�=�Ԏ��<=�ʻ��{=>1�S��<Q���<���QE=�^W���ƻ�ݓ��׽<y�]�Tt<=:���,&�o�λP
�=B����=���k�<΋<İ���m��=q�����f;�����0~=��m����=|H���8�<�,�<e�}�h6��.�ɼ{/<�����z�$:�9��ڼc�p;]�/=���:����"a=�N�<�����<�rƼ��;�#���C����=C���C�<𻽐}w<�BJ�+!�=�i�0���a~%;�X�;����(<��x�d���~���U�=�%3<�	��2O���l�<���>�(�<RkP=�[o��)�Z;F��:.=L�K��ܗ��!d�3�m=��-��9=奚�3�;�L4;u	���T�������͑�=��7=|ٌ�163�%뽿�$�{ǡ<7��W�=�Õ��d�=<�q��dd�zLn�T��="������<���F[�=Sڅ�2,�;`�z�R��& ��M�j=�%��j��q@���m=.�6��h]=:�xκ<�!;�P��=���"3Z�m1D=�`�=E=�seg==c:���<����dB�=3`����=��y<ue�=�D�;�K<=f��k+K��:��ſ<�k�c��=@8�^R=9^ǽ��Z<�i���=�G���<;ħ�酯<��`{=��ӽ`��;� ��T=�m�����=�)���$=�s���=����=XK��Es����%�K�}=��%�� >�z|�^��=���ȲI=�;�z�<���v�=�F��R�<�v;=����O�9��<�4=���������Y��û�["���ܼ��ܼ�&�ь�\ �x�><��/��ۣ;���x�<)Q�=�b��V��㈽#�������h>T?��yͰ=#cn���G=�~����<�H轮�	�+9e��5= 3�9�^��#"E�׵�=�����K+�WFy=Y�����zCI>Qn���@�o�==�n�=+\̽��f=�����<>���m(;�"�h7v=�U�F((�ױ^=�Ƶ�#Ȋ�U�}<��=wJ�i�= 9����<?�='�'<o۹<�w;�f���h��V�)=���]k�=i ��V�=}��D7;uq�����;�P�X�}=e�ڊ��� ���<�H<�iM=��+�uE�<✽O��<%~p���=�;vZ��E:�_d=񎎽`Ƶ<m��(=�W׻���=Ǘ�k	��f="yW=}g�<�x�=���<[�k�m��=���=��T��8F�<���=I=�^�<u�T��e=�W׼W5=��m<c���Ί�<��<� ������8�5��<�f�����=��������p?4=7�
=��<�? =a�ݽ�c�<]��ď���M=_�v��&=���<ń���:g<sm����<ꇂ�3�=�&�<5�<<$�<��kU0<�w=���:\�~���h��ɖ�<�4�<�v$=��<��=�fj�<I�;�;}�<����R�~��</K��D��=��=����0RF�/7=2#<=��.=I�o�IpK=���<\��;��^��[»)�;���=}c��h�<�՜�>f.=�3��-۽1j��<|�Ѫ���6�i�">_h>o�#��"���=@=�k�<���%Mr=PY���)=�Ep��?=yQ��P�=����b��&�*�c���{�2'��}�5>���=�h��OU��cݓ=pE7>(2�;}9A=|�k�H��hL廖>�v���հ��K�<��=�� �0Hs;����G��<� c���=�u����Bf[��ɒ=�n�څ=3���P�����<� =0�������!��;
�S=�1Ƽ��=�oȼ~�B�^��='4��E�<vW�Ɗ)�|bo<���=��ֽk�v=��/���=���gӨ=� ��2=�\=��=<���=,+C�
3#�Up���Q=)h�<�k�Ά���z�=�S<[�=!���a�W�F�����:1�h��_R�����=��0<�	�=����F �� ռ,@<�=���˽=(�:=�S��|jc=��<����=F����=���<z󛼁u���;�/�5`=�t=�~�<�j{<͝Ҽ��o�6���Y�м�ͪ��Q���t�<9��<���<q����ܼH|v=yy�j*�<�X�������1�<kH��N>�佞��<�0���Ȃ=Q֜=��}��I�=�{n�Ɋ����Q=M���$���%��-�y�>�Ǣ=���ȟ�=���y/���&�>;w����KT��v��=��x'�=z���+���@<��<w������=��K>8*@=�×��k>Q�ƽ�㛽0�]����)�#�F�ý������=��T��=�2��d*�=�����B�;aL�<�<4��=|W/�~��;k6���i�L]��6伪Sμ;���Xo��cZջ]mZ<:� =�|�^tj�
M=��d:N~8��=I=�S����Kڻ�ż��Y=�;�<�M<A/T��f	<�4=%�=��%��i=�	��]6�=�	���M<^h�-ʴ��e"���9�)z=[��<+��=�鼘�N�o�=j&+;�~a<��7=���M��
SM=K�Q=�9=�|&�Z�<oO���<���<G����<�����=�Y�^ _;�<d���P-}���	=��߽-~�e!�<(�R;r�E=�tԻ<H=�Ca�ݖ�:�*@�x���E��<��=��P�p�8=�~ػ��ҼN'M�1�j<v�D/����<��<��Z<c\мv�ü����0�w),�e2=������r�=�@��LP�=FI�=�����Q軸�<XFl=�ë�E/�;f";Dy�!�*=�"����=p���<e�<b|e<����+T��8�=�˻�↽n�b<��<ޮ2�u��Ӛm<?��;ӥ=�����>��=l��;�:%�ͦ9���=�y������D'<�n �ɘ��ZP=N�=���=��ʼe�==*���I>Lx�i�<>�9���7<]_0�o�b��`b��ѽ�ux=��F=B��<KQ=�%��գ�<�5�b���ѷ�:���\l��m�<�@=��Ӽ_ox=��<�X4�$Eb���=������(	5<������=1��<w�.���?9?B�ؼB<�E^=�c2�!>�I�d=�"�H)�;�ւ=��s=���V^(���m=��m��Q��hļGp�<b/5<dDɼ�5��
���U�G+=����`W>=���<9>=+�+=?����˻n�\<��=����<!�#�di���]�$�]<�ݥ==����@��Bd�<�_�m:��a�=�Q��h�=�w<#�8��Ƚ�2�9��: � �X�D��<��1�K�@;��j=+\=�P.��5�����7{�M��_��<�'=s�U<-���y<��9�#W�P�׼(!�<�3t=�S==��,�=��<�ե<ƨ������c��μ2�x;y�?�K+W<z]�:���9P��=�lh<vfȽ�>y����=�w��ȓ���c�B4n���;	��="v=�@p�Ϗ�<m#t=�ѽ�2�=޽�<�|�<LGS�~A�=��_=Ѝ��f-Ƽ�:t<��$�/,r��r=��<���<�E6��g�������=�eĽ�O��9��m��~R,>�~G��# ����=��=�E�=����AN=n����<�T\��$�<�FE=^�Y����X�b<��=��>+�a=h ǽ�e���;P�=�F�]ҽB�=C�׽��>�l2=[0��+�=�Q����X���1m��o/����i=w��5�k�i��=!=G&>�9�R�����W<���E��=��u=����BR�������L:�ʃཾ2���U��}�=lx>�	����<�(ؽ�W;��������B�iI��k�=d�=HdN9�����=���&J���˫�&�d<9.�9��=7�=p�ĽR���Ƚ�>�]ֽ�� �<���[�<.�;�`X< 31�4��;�kZ=����w<>=���<�8=q�=����y7�� ��:g�����<�L:�䇹��=z+�����=��(=i�Խd��=�tC���,�}�Z�������d��٠<(��=��<����>���:o2���k���(��s�;�y=d�<�,�=��7����;��=`RG��c�Ҧ0��6b���D=��o={펼yۼ(]�<dj���^\�=����N�<݀R<NS><�����g=6��WT�<:{�<T��_�:�7�X\��''� ��<C�tm�=�3��n�3:�eB=���:�!��j�����=�o�=j����1��!#�<n7���½�㳽B)���o۽P�C<7=-"��6LO=Hء<���''�#V׼x�<�#9�9�I=�4��F��<�.R��y�=�˽ك�5���sُ=��	�hX=��˽7�ؼ*�>�Dt=�w0����=lJ=���<�k
��+B<���H=k���<����>�}F���3������h�t<f($�J1<���}<=2���j�<�(����$=ם���=�jU��l�}W��ɺ�6*�"Z�w$�S�y=N�3�ȼ�Y:�L�<*�h���4=^赽�8���(���-=d��C�:V(�<��@;��ɼ;;��p�55�<0ʇ�tD�<O��;q2=�ݼ�L =j�L����)|�<�X�<x!=��D=����7��=�-i�"�=��I�a��=��ͽ ����)�^T�<5�Ͻp��;9+=��O=�<&h�Eӂ;Q��SmY�0��q+�<�J�>z�k�<;>�'�m�=z�$J���Px=��d���J��A<=i�����<�=�o =B��:_sW�4􉽤� <�"2���>��W<d3�<���in���E�k=![�<���<������l=B聽�s�3�=��<������!���L=�NY=�락���=�c�<�d�O�����1<*�W�kɀ���@�K�W=fc�EJ=�M��	��]ձ����=�i�0�}<D>�:]0�=� �5%`=勱�K��s���� �=B_���=�r����=���J�=R�|�6xӽ;��+q=z(1��\�=��E���= {���;<� ��e=V��f�=��c���=�û=y\=Z���ί��_��Y3����:7�N<#ֺ�+����;�=����7=ֵ����<���ϐ=�Z�����=��h���r==����=��S���m�y!�P>GΊ��5<�iK<�9�� ���5i�h�<�TV���B�\#W<���s<�Q��x���榕�uL<��	�40�<���<��<�е���=9y��=�ڼ�W�=������9Ǵ���=e ��Ռ=ٶ�!��=�ɼ�
?`<�ak�u����f��}�=�f���g�1	�<SR>lq��e�`:����������c�=m>�(9��s���=����+�]��0E�s�Z=��9^a=�@��Ք�<��<�!L���=>w���:�;2ͧ<�w�<���̤�=fRv��<���=j	�|9�;|b>���z��K�;4|<��w<T��=��G���=I"���;�=�k�1=��!�u��=^�u�t$y<�� =e9=��;��{=諹�m�!=g���W6=1,�;\��=�����Ld��K.<��=L�T����䃿;�S=�*k;帘=X5y;%���>�Q=�k�=��l=B˱<��R�ʯR=ያ=����RP:����<r�=�d�� �Y;{�׽^Y�=�z��_�=�A:<�ϽF�q;+6!=P0� ��:"�@�@���x��<�==�����ҽ���=ld� �;.��;���/��=��a�T�#�4=�;5��,�=����z(��)��G���{Z=ߑ��$9�=�O����[<݃;X6-�'8���&�=�c?<խ�I����|��`��;���<��=�I��0K;;3S�<jX�4��<>=Ԇ����<�%ؽ/��<�=��ϼ��c6�%)=/��=M4�<�ǣ���<�fػe8-=!em�l��}�#;�*�=��ڽj{�=MR��;6^���-��=�<�1�ko����=QS�=ȑK>ru��tؽ��=؋O=������<�s��9��=�&��QǷ�4r=�=f���!`����+;�����S�0
=��>魻=���Ě���M>b�I=��=ZfN=s�9�$�پ�'f=�u�����_�<]�ۼL8<b{Խ�=+h���.=�@����l=N���4���r�=�T =�Ò���=�W�<T�L��3>=8�ܼ��B<<���V�gb=��H:R�=׊=��~��v�=
7��U%<�	���?����;G5H=��9<Ԁ=pmݽ�C>��/���=n�߽4)����<�肽[��<���=ԸO��\"<��<t:F���<�X=�1�G��=lh��q(u:�w<���1�5�:�<�Ӑ;�<�J��.��=��X���P<s��ƂS<��ʼ&l�O>=��=��_���0<�X��BWƼ�=�Ϩ�?P7��д�^c
�q�<\�;���<8�w�X� �GB�)�oUE=�=����u��Eͯ<'f���#�
��<\�+�ۼ$ɼqHL���f:�z�EN�<@�=d�l� k=�Z��<㼑�˻�C�=r�J=D$����U;�'�2c���o=T4��^p=�*ԽV#<=m�2>�1i��3<�>���׽��N��5b�%TǽS����i�<�>�=^�@��O�< �.�B=�<���&����*�=��)>V�H�F�]��ó<,�����th#�D綽^>��l���m@=pa�=�4G����=�T����W>dV��I̼|�ν���<J�=�f�5ǭ=�B�Vd�N���i���C��!��j%���An<��C=��;F�[<�����<!|�.�ݼ5\C<1|㽹|�����=��*��M�<W�/=j�<��=�:�<)���x�<�h�c�>Ȟ����<N����=���;NL����˼PDӽ�	�<uo<(_=M�l�#�b�ሜ=�	�3]��-�=7��)��5A�<���8��<�٬� 	�<�K��)r=Y�G�<PC�;p�C��+l=ۗ3�gL�e�*<8���bg�[7�!�������!=H�<�7<�Ag�<�TC��}=j�	=g��<�Lp=�"L��)�:j�;�,����{��:ż �#�" 1=w���=﨑<��(��ݰ���"����c�．I7�T }=7n��n����=lvY�t�=ڂ<<%��<�E�<��:�4�|=���=���f��M�=AE>;��1����@6=<�Y�CѠ���;�������=Ds"<�`<^=�m
=�+;=,�#�<�$��pe=�?H�Z���d�1=S���,���2�=��Z�;Խ��<���A0=�mֽ9��<Mg�<��i=�� =#R= ��f�`>��=:�>Λ���b��6�
�� a���}�b�����.=�?I���b=����*=<Y;�|�v�5�}��=ސ��$x\=!d�<u���v�8�h�=1*�<|*[��V��*>=OD��Y <�=]�Ľ�õ<)/�<ib켬�<=������;m-9<��<�5)����jFw=�νr�<�:��m�=$ǀ<Q[c��|<vG�<�>A�f�0=w�
<S�"������t�*�L=�A�pH+=�h<������=�E�5�;Z[��"����`,=�� ��Z0=��iQ=àd<��=@=�<�ly�}�Ȼ�m�=F���Ĉd���=_o-�S�<�m�<YX��Y;%Gӽ�<��<tj6<A��;p�z�><� =����II ����1ٝ�+e2�V[��L<K�=�b���ɼ�����뼗��;
l�8��=[Κ=�@�<�9/�c��<���4Z��ݭ=T��b���;�7��Y=~з;Pt��V�;#M;<ڧc:�p���= �;��4ƽ���=��l��S=x�<��<�<4��<�8�=�Y��V=��;���ܼ��Y=�Y;��;��6�n�.=�q��鈽ی=�-��+1=�އ<�=��o=7tἪ���8�<fj?�&^ݼڷ>��%�-������=�H=z/o;�mϼ���<��;C����c����l;qm(��:<OG��מ��iv�h�I=��=��d�|'�;4t�hgd<U���b�ڽ@_<�ס�m��=��=o]��6pO<�<�R꨽?c�~e�l�μ�Vp���M=�(:��;�=�=1!ȼ��
��,d�ʜ<g�;�#���f=�=mvd�by��è<"T�<L��̗ѽe�;������=u>:4���4��vg�	W˼s]���a����<<����G=�v�=����!��t��<.lh=X=��j��y_Z�d�B�P��=�<�=�k�������̏��~Ƽ-j��ap���R2$��ĺ�F=W�#���#�s�܁���B�7il�LM���;����7:x��T�뛐�w
޼�P�]����;���(�==���==dk��=�z{�~��)Ɓ�S���CM�Q����{>�M�<��0�N�������OP;�O0��ܽ*�|<} �=�7�����w�	>៽U��}�<��<�PO�Rk �x�W��<Qw�=�95���&�V�<�z�=��%���罾�����݇��������;i�:7j#=�`���T�c<�\�n�н5'l=��?=����Ӻ=J&��>�����o=q;�E=b��Q1A=�=20��ެ<=�I����<�֕�B�v6�<��L��w�==ޏ=�%����<0ڋ�}cg�KY�<��߼v�<�ۃ��b=o�ؼ��/�Pz=�64=�˽ۺ�;&0<め=.�6�t�=�ʽ�a#�Z�3���$=@t�F�<r������IK����=;��<��g�.u2=���*�:�l�:QQ�����^G�����=����W����<�
��*����%,�
���������m_=G�6���2<^�<�GD<�����bP=�������	,n;ܪI=����A�=K=<mu����<�w(<|.�s�<���:����9��<�&�;��*=���<���;�X��<A5��=��<�1"�-�<;�=�E��Ѩ�n�ϼ�*��,�f�=Y����1�����*�=�N�IH&=U�����;m�]�kټ�Bu =����'�=����E��~�<�<��N��=q���?g�>����6��h�;�e��>�y��~���<��~��j�=9%m9*n��O�|=�)-�'-ս�Q���SļM�<�-<��u=�4�<#���RZ�� i���=�d���<>:�<8����zs=�I�d�G�Z#=y�i=+�K�t��<Du=/��<��$��<�<*��	��<M=�"����7�<U�2�T=t��@��=�9-�S/P���<�ߝ=VŽu�@�ºR:�6=��+�/�+=�������2��	��=�6�L-�=�����>T�l��=��a[�;�T��_�=d��i��=Su;�P��=P2<'ц<�����мI<��Vi<=�ٽ����=����Q��!^�𬃽�襻��<(��<���d��H�&=*/+=c�̻zm=xn�;�:9�K��n�<�P
�6
=�c?�_��=Z	��=�=_i����km�54:=U�4�Ţ=q��F-=ڙϼێܼ�=���;���<�9��v�n�9����<��=�vs���4=�N;<#�<�_<���<�%:�&�(���H��<5����=u�����W<��ʽK�q=g	N����=!�Y��ݥ=�닽�a�=����[2�đ<5�<uʽ[����[)>��E�-Ƙ�4ٽ������=H�1>���=���YI;Z�N���<�����J�`�E=���< ��\�������w(:l�;>�ݼ�~�8ď����<��e����=��˽&-=�`>����N8<LȽ�7���Խe�m=��.=���=��$��Lw=�N�`L�Y[���UT�:k�1� =�����5=����5=q��<�Ȏ=�92���Q<�����0=�Q��Z�=g�˽i�U��s�<��0<����<�T��7�=�{]<�bۻ�{�<H=���~>��<㮴<�y��^�<?L��F��=iH���*Y��t�a�=��<;�����<'�(�"�)=k%<b��&�������<6���r����	<I�S;3=��w=��=u4�!�P�-7N<��F<�����:d#��Ab=��@�C_��L1�c>���)�Ӛ���p�q�>��;���w�<W0�<��h=c�l�=3=V�(=d3��H���=���<��#;eC�������H�<�ú:��-=c��<���<�69=��_�`�BU
=H�	�����1z��E�=EH=�P�(��<�k��1[:�%<�B4:*q4�K� =��b�U=b�%��R��z�V.=̳��,�=~����u<
>b��=^?׽����4�U�&�=j`!=�>�+����,�RZ���������ٌ�_���3=��2��^���=�:�<a�m�ϻ�����P2��ŭ<Nu�=<���㠾Vs>�O=�N�<�a=:�	=<1���<����%�9N<���<��=i`ƽ��/<���d��=Cl��0%��۪�������<X"�=�:6��S�<�</3+�38<�aݽ���fr%���>��ॹW�*�Y�=V�=�����=��ѽ�����&�,��;#�彼�=���;��(>���T>��!��r7���q���;+�==o�Q<��$<+==7^	=���������d�J)Խ�j��0.�=�$���(=�9����-��ҝ=�=��²� �6�0ס��T^;�!���<h�Q��<� 0���V<�5��5����:Ta<�a�<����<�=�J��K�=�䨽��[��u&�����餥:���ی�<��<lO����{��vԼ2f�:��>=�4�>Ă�v���!G����<���<�t]����<'��<,�=Aԕ�5��<}�]=�����=�#��y2=���(�;̟��<we= i&=� ���.Y=�s����N��X�:�;V���R=��}���!>&ן=����!�=��U=�\�%ʷ=��ɽ�����ͽ��üi'=1MC��`�=���d��b4��g"7�_�.���:����=Xi�;� ��Fd=xt�==��\V<�T�S��&�½�R��4����<�����2>!��K >������;`:����_<r1�=���2��<>+=Gy=~=p������kh�<^����X�<3����<)��<!�D��'��?�<߫~����E�z��!�<eo��A�=+A1=gj�|#�<���<�N�@Q[7~�X��h>\-+;Bw��.)��֔x= ��<��ٽe������*Ӌ=8=�O=�w����@i�<�aս�Ǽ����̔��e�������u$w=t��;������;�64=+�f�/Ą�`�Z=&rļ�O=>}D���;��"r�c7��ӆ�>ߺ;C��ᰙ�S�B=�3���I�=���旰<�Z&��U��Y(���F�c��; �9�a=L"=�P�<��<$3U<���;H�#=��¼e�ͼ�G�����<: <@]�<�}-��)��v^�<|l=��%�I���!�=�~O�I x=ٌG���:���=#��(-=����M�.��?�=����+Z�=|��=���<��\L���:�=a�.ħ��<�ay��w�=��8=�4\<qm$=��K��ux<�s���;A�=�X<<B�==M!�<g��;70x��@A�%+�;h���b�z�K<�1�� �=�-��#�<	�G����<��ݼ���=��2��� >�d%=�=�/w���P;�S���6�Me�ʘ��X�;�Z=��Ҽ���=�+=���;�����<��<s'��*�=`9�;.˼#�S&� �����<?���T5�<r篼�̪=���<�8����z<�z��|�ý]���A�f[U�ᡰ<׵'�az<��:#;l=��/���R�� �<���Gֺ�_ͻ�cW=
l+=o�:��s�G�=�kA�����z�E�|%<	�!�<�[0<mZ���`=V
i�m ^�G�Q�����<+H9�D�=�%�0~���BN��
=�ļ5[���9�����%�<�{�<oYF=
�7�N�;̗���3_��;΋�ͥ��A�=e��<>�'��o�󆢼w�<�)��4��<��<����f�;^�m��L��N��ȿ=ηۼ�z�< ����B񻖘��>�pT;|�(=j.��ق=_1��ĽhC=!u�Ŝ)<̯<�'�=|�=Vi������7Ҽ/|s<ʓ<<���aQ�<SJ���C�k�>M��s?��}��2��<0N�9�/��XH׻qb�<yQ=�V��Ih�4��R=��-�' ��x@˻`,
=V^_��b��6V=�oG���h=��=*�F�r�<ES�<Ij�;坄��,��Ha�x�ܻ#��=4�;_]F=]�<A�c=��５���>�<3<��Z���wI=��<�ɷ;�1<Me�+
�|���\>^%<��ͼO@�;(C�]������a���0P�=�ə����=��=�7����=�����.W��?������=q:��k=� I=���9��Ң9:7��������;���<��<X�=/ܘ=��׼� 
<�8��,��5���۽o�`�A���<�3�=��ͻ�;������0�����C۰��΍<W���b+=�<�=�V��R=悢<����n��iy��"���4ֽ�W�=�ݝ=�z�hċ�a���N0�<��)������ݻv
�fב<q��<�kӼ��S���g�6�I���x��<�0=lF�<���<�� ���$=���� �<�-���;��;<��=����T��<KT=D�4�ޢ�<��Q͈=t�򼫦˽O� ���7=�&	>���kb��3�3���\��O��������U=,�<���9Xs=4��h�gڻ�9:=Z�;YS�<<e�<�=b��=z���Zd�6W�=�3�J���]���� �>���΀=<|ȼR�=� =o <�'�;M#�d��;�!*���۽��:��V=*N%�iϖ<����t��<�3�<i�+<T"U<LO�Cz|= l�����"��<���>�ʽ�溽s
5���<�bݽv��<��={Ǧ<��Q:�㔽�r�<�C;5��Zn"�\����F=3"�<\e��8�=��Ƽ�l���4<�'��-��U�:�P\<�vT�%�m�7���:�#���=N�<��<ѦͽBi�=��=�'���;��[�?O�� �<�t��e�O:�9$��*=~�<�NA��,=WÉ��}輥z-�Ҏ����!=뮽��<�*=Z��;0Kz=���;��W�x���(��ނ<[�ڼ]�=<d=MȽ���<��ܻ������=��r/=J�=�(ͻ�r��=��<af�<��t*��<9�;�=�=��;��k;��0�O!�<x��<��)=H��<K���;�󄻲����͸=�U=;cԽn��<�X���B��:c=�Y$��.g�8�7��!̼qmƺ��e�QkJ=���=<~�1�=�[g�ݺ�=�`��K�>a��������[��Y�;��=���\�b��Z��Z�輬��=��=�Oc:)=��U���r�s�l�}�~��=��H<�T�=���L-n:Ao��<�E�=��׼����*=Dn��-��*�Y�¼Π�<�P&=����R�ʹ��9���,=������=34a=��.�U�<�Ҧ���v�}��;죸��f��0����/<_��WC��6��<�Sm=�û��* �������<2ź�Ƀ=e�۽�˝�ĵ�;�=��g�#�=�]�W��=`�ݽ��K=f�ӽ�r��o���̿=��(=n3�=�޼ W\<���:��<璅�2mO�f`�=i,=��e���~��oټ�UG<_h4=��7=nF�k����뉻�G�<�,ȼ\[&�|���}S�<2!�=:7=էܼ?����R:{v'; oW�iw���o���y =Nxʽ�_�<Ȉ�ڹA�ؖ��J��=@YǽT�;�Qܼ�3<��s<@�<{;!=�=������<R���<��;�<��}<�w�\�<���<�Jt�,ܨ�u��<1�=��g�=I���>;�=���G��V89��B�= ۜ���=��ν�͓=��ǽLNZ�lh =�����=�=�G��l%=��l�m��=lm[�0u���5����=�F�#���%�==�*>Fߵ��G6=�;�7"=�=�ר$����<��=����g����>����sj>�r��/ڼ���=߈�;_yk��R�=�E��Z�l=A�=���C�o<W���)�;�G˽IO=�Y�<�=H*��r&=�B��j&/=�XV��� �b?5;4"c=��4�>s�	w��)�=�`�<D*k=<�n���5��î<��=��A��36=rB���O�o��<�:�:��y���AN��w�<>���iSѽ\ �������G>93����?=8*�<��_=ʨr�P��=��Խ��<�O="Jm=��=���������;m�k=xX�<���Ou̽�s߼q�<d_0=��!�cX�;ck�e�V=�b =#Z��m���臺�E�=Gm<�K�<n�^��<��~=I׼[戽��;�]���%=ɒ`<pb���;�⚼�M�=�<9��c.=g���Wռ�N�դʻ����:',�ç$=��,�߂��Fa[�G�'<ܿ�;:�׼=�q:Kc�; 7�<�?�<������<*e<f��=���=5��7�<q��i/=F�A=��D���<�ك=�,�ƞH=�ѻJ�]����;À�=
M�F�;P'�$��Ҫ>�o;�q��3>���� >-8���=���������=��l=c=���ȼ=�张�=�>u��Y��xL̼l
� �=��l9�Oؼ������(=��n=������W���R��>k�k���<�]=�0�=��ži��=��K�0�=�9��T �<p	�=�Õ���n0�<�=�Tc�M��q�q��vv�=�����i�~U=���<~M�<�+<dt��@弬 �⻠���Ҽ�fn;�I�<�=�=|�z���%=y������󁚼����k�3�#I�=&;-�:M�=IE_�Y�>C�F���^=��$��<<G�<<�$�<��,=p�<�=����g��|?g����pm�=P�:<�g<Q�=U��<��I=�*�<G-*�TƧ�A�=��'����=�"8=����Q<���=>���W�.V齠�ۼ�b=� <b1���L�ic�=o,;M��:��g�Y]��fw#�Z�@"}<� ����a��z���R1��м|Q�롼
��,�<�Y�%��<в�3U=�d<=P��<1�w��
<�.���������u�<�=:�'>ދ��v�<l#����h�"�߼���<�2�:���<;u�0 %=� ����<�K��O=?=IJ�P�\=�⣽�ƽH�$=Uӓ=ћ�<� >^k&�{�=��U<q�Z=o��=H���%��� ;��{=-E��U��$��h�0��&>f%�Z"P��P7=Ҭ>X�˽@��=ϕ��Ċ<C���
���g�/��!�=t�h���/>�н�[E>x����0=1����=&�N:{7Y<�`�;�|=f�<�V{<Hfҽo��dHb������#�ٚ�=�x���e�;��b=��1��Լh?a�V:�8sԽ<�Z�P=ec�V�=+YM;�N	�S�<A~�;{��I��<�ף��u�=�x��o0�􅟼W5Z<J��;ܑ;�׈<�{<���ϋ�<�A�C;��l�=0�Q�ڙ	��J��ִ�=�<�/��2=􃩽��<�Y��<w��<bFn<�h��� = ��;�����S=�v:ŻQR��M����>�
��:�?̽4��;|a�=����S�=�P����b=���#��<OaȻ@-�<��+�d�)=V�%=,�y��KF���Ӽ&�o<�3�<8�۶=zA�aB�0�<t%�<��h��z��<[�H<�F��r�.����=�1�� �=d�#;0����1�=pJ]�j�]=+n��<݆=���h�:<ғ/=��Y�5i=!0?���;c 	���W�^3�=��4��?	=�
=����Oa=�t<|��:mY�<�><�G=�|ϼ����+"</2����}��J���P<[����~��ټ=)�3�B�=���<��v<z~-=��������ly>J��U�C=�z8�C�=�8�d�&��X��ы<e=�8���Ph=�� =���ԍ�;*�=�r��	o9�s���l��H�!�W[�<0�2=_R��v��<z缧尿$�=�t��t0�	qP��;�=�0d=>�:�,[�=��2���U�iu�=�7�䧵�Vu��ݼ�b<�<s��<ޤR< s����<`��<��x<�K|<�����LH���S��==�����<��<���<��׻݄�<�6�<*�ǽ�H�<�2&�3�x�k��93��}�� ��<*�=V��z�S� <�$�T|5��j���4=��`�k~�=7��<>,;=^��X�=�p��|��UP�=�L�a[����;�*P;l�2=X�~c���V=�q�����A �'�<:E���=���~!='�鼽��<�󄼏�	�.���7�)�)>
>����oH�=b����Ŏ5�T�����<�N,<�g�:����f�=۩�<;f��Ε:�k�����_4�<��w�4d���;�1���=Oc{��z�S��<��½j<�=���<�4�0�X<��~=-}�|���`2�=���<-��<88����<�0�<��cɛ;ǎ
=��I��<lP��os��9O<�J�:\=ل0��ݛ��d;�$�e�	>3�<:|�=c�;�]�<;�D�(*q=��%<�r�OSȼ1W=F@�<?bʽ���<�T���6!=�7Ǽ�P�=�w�߅ܽ\�=��T��57��T�����ʬ�=wD�Mz�=�=mo*����;|;�O`#�H>1�4A�Y�<rڽ*=�:=��= �<�L�<��v�EwǼ�eQ��8��:t��=��R=�#���J=��F�B��Ly�#!��TJ�<Ђ����=���<�/_��k4��Y�g���������V�<bj����g=��=�r���@�<�ʽ�	�j���2t����%=.7���Z�=,��=�T�<�"νT~����<�������2M�<e��������<�b��X�3�|��;^G�e� �㻘��6'����<�Zi<I3���<\��4��<IƼ�S��{�<9�<�5�LZL=�#;=���<!o��������\���n���C����=4D =���~���Gz�<�LZ��r�ư �_=vf^��mۼ	�F=9@<ˌ��o�b�i��=^����p���Y�c� =��<^7^=�j=g�����!==P(��F�������@��<�/o�ąL=s��y�D<j�7=�n��׃��Ir�<�
ڼ�ۃ;DŽ_ˆ��	�=��M��N�=���y�-=�C�:܂$��D=G^=�`�=�Q��^Q��Oi=����`���6�N������Ӣ���8=I�=�n�� ż5�;��\޽��k<�c�������$�E��;&��<��$�;嗽�{��i�=o��	�R���M�ջ��=<L�`�"���#$"�pP���	E=��?�y�\=β��|$�=��_=�/��k/=4+��a���j��!��|Tb��(�#D�=|U<��P����)q��sη�a!����O����!��ͦ3=��=^���`k�<6E���KZ��{ĻΨC���ٻDK��X=���=�`��9��𙩽"w�#"�<+g��x᯼�
=�s)<��.=�
=M��!�1�:��+<�&ͼ�#��c<g�<�	=��#�!؛<O�i;'~�ӂ==ÞJ:���<�)/<���<;�<�꽍��<�2h�����\/�=L)M�8����w	�Ve_=瑠�}���^=�]ͽ$Թ�ۋ&=v��@��=\��=��N>H�@�}I�+���X�C�=�����.��<j���������=:{=���̸��$=��
�ڏ��yx��+=	;Ћ�=`�>�j�O�o��)����<����<��<��LG��K{�<�N�<�C�o�<�i�����槺=���;�ɧ;�U^�Nҩ<l8<�ײ�?�.�5��J��4ĕ� ���*=z�^�>�=�b=';����p=�F�;�x��HH=8f8�7�=u=�<<�׼����S�N<��=����?l��N(d��Ѷ=�)E�Ra�<�I�I*=WDO���=�H=�n4=�V�9�0�;B3�<��=%��<�kҽ"�&=s����.0;�xM<Od��k"1<���=���<� ^�'���=�X�;��<��Q�2��JO�=�{T=�7�=�д;��ݽ�#<�y2��憽œ���ٽ��<�n����<R����5p�����s�=8�'���v=��ҽp��<�K=�����_�&sk;[�ż��7�Z;�F�;#��x�P���[e���<�P);,"��&<�d�<��<)=pe�=�c=F��<������Q����<���=U�D��G�;־��Ҝ<Aʽ1jP=��F<���*G<}�3��E��Ph-=���;� �=�-νm";ʹB�-�3>�9i��d�+(ʽ&p=d4�=��;m�Ѽϸ�<��<�<���k:�Z=;�|�ժ���Yҽ�&���=��<�s(<A��=��:}.�~!5=�Nǽ�O\���l<VbR���.=ݎ���<|�ݼ�*=r�����T�Xx*�i��;�s@<��N=<��<#���1R=��6�pD5�.1��(Hj=K�1=7:�;s�5�,c��a�|;�`�=�C���6����M�i���Ze�=S�m���C<��ȑ=�n�=�m=cAڽL�4��Ė�1'�=�Ǡ�*��<u75<QI�;����ʉ=	��n��=�z�;04]<��K=&_�<�ҩ:���=�h�<��,���xm��1��=�HG=�Z���="g��F��=E<{P��5%���]��k����==*�C;ֺ�<Z5V;��N=Q�<D6V<X���_罘���U4�=�U�;"6��̣|<�XC��KV=��=4)��0u������=�;��'=w�����<A��;bK/=ū�[�'=������=��ټi��ӈƻ��;f�_<ł4�ZT�f���vں����Oc=f�����vRt=�Ū�y <�\�=��#����:>N<�c*=Uꇼů,�Rα�z�	�]_/=���<n�5=,���ǻ�mm�>��p���X��<>Cg��s��=1�=���Jb�:gj=�[^=��=�A�<�*A��";�Y<`��j����ؽO/3��Ae>�/��k�3��X�9o��=��K�S=0\����������{;>1�J�8	>��=��{=�^ ��a�=�W����<�-�<�;�=�3�<�S��\o�b��:z'�=��w����9�齥9b�G��=�[��;J�<��ҼLmc=o-v<�]=ɞK�S�u�^Ab�'����u����<k�b=��*9�߶�^=��$=̅��$o=n$�cU>e�=���=�bL�_��=G`}�]��:nU(=%�<paK=��-`�$��<�F�=��������7�w���=��d�e�&��؊��ޗ=!�=8��hF���<�K�D=�7J=�����J�;�?7�^/==�Q<9��*V=�����6���;�෼Kf�<��q=��=���<A���R����¼ۛ=��<i�=>�� D=W�
=wP_�F�%=>q�<e�����ݼQb;ZX�+�G=K�#��.Ƽ7W
<I���d{�<�����dC���i��}=z[=si�<�vϽ~��w��j5z=��Ļ���>�ݻ�\����<]�m=��<Q��a͏�Bk&=J�>��'��E�+	��X٘=~�= �:�ٸH>���J�=6A�;��4=IQ��	/i<K��<`==m�i=#P���$7�6+߽�ѽ/�T=��q� &�EW�=t�=V�S�;��=uhн^�=�pŽ�y>+�%�߳�=hny<�>�.����H>�Q��X4F=ښo���u<C�=9����߼�B�=&v�<���<'�н_cK�a�<��<:�e;��=*�;���=I�=0ֹ��q����B���#� ®����扢=*��jU��n��/#��)
=-g�<� �o?�����n�> �e�R=o%���� =v儽�1�=��I��]�=]�1�U�P=��'D(=PAW=�"���UQ�*����q�WK�<J�/<%>��������<���;����� ��r'/=�T��9u>q�<I���<�N=T��<�2��b���]`��	��J�.���<3-�=�P,:�� =շ}<޻���	��,~��b/{������<Ȳe�=�E=j���̼���;L:�~H�<\��<ڻ��A=>��<_D��ˉ9�]4=�Mϼu2�IL�<h�#�|��@�<�>�P��
_���#E���%=�n����Qǽ��!��v=Z���h<=Dݪ<>���l�=�=���<+�,��W��vG�=�f�=��=�̽���;	�;5��ƒ!=�5�=���;�ռv��;��E<C�7=�|�B�5<<0��
���\��U2ܽ��=�z=0&=I=IAO=�<���1=�2��?>�/���U=�[a����=e�)��C#=��)H>�~=p~<;J4=�E^=�͸��
=t=Bv��<9�$�	��i�;}�<7�Y=�z�<�GV�SQ�=g�}���R)�0����������<)>=�c�=����'�N������N=w��;�j�F�t��.`�a��_���bd=-y���И���F��6�s�)�b��={3'�n�s��}1���=�������)7=`T�'ĽfYx=�U=���<�:�;�=�Ξ������<�gJ<.3�<���=�Ҍ=~D��x=��'*=�#=J��^:����;N\i��y"�t�?=5�=��S�X�M��Y��T߽.Q�= S�cx�%��>����wj={�;�z�8p�9|�=70V�D��I������:`�%<����$=��<<0=��=B�;,P���J�=��=�ԻT6��C ��'Ͻ����<u�=��\��ς<��<��<���z8�<%�2<�m�@iK=]O:��J���k�=H���֡:�
�R�����>������	s=���<0#�Xg�=�!9=��|���B���=�`��� ���4�(;pU$��S�=�d<>�K<,��<m���"P���7�\�����V9<e���ؽc����<U.=]�s<�Y=�I��
=�%o�iv�=�ƈ=Z-=�ʃ��pU;�!�������'=J�<7A̼U�<��=��h=ggս͂9=�g�`X4�|���ZE�l?�=�`�;��ü�Pp<�����/���\��벗=򋄼�x<r�!�"+&�a��<5��<�x��n����Q���;}�ü�ּo�#=�g�:t2=�ˇ�� =�ߌ��2׽�C�V���q�����&q�=�Ϋ=ȇ��t�Ƽ� Ž����f�����T���m�<E� =b�o=�����3=�6���������<|]?��F��=���=v1�=�MŽx p���ؒA��'ȼ�q��M������ɥ<��Ҽ�5g���P�X�F=f6J�\��<\�������D�<Dx<��C� /�;7 �<�R�<8�A�J��<���<���<ƺ=��;�z�<������,������ݽ��<Ko�Ι�o�1=��=�p1���P�n��;�#V�VdI���R�P$�<�)<�*���KW�B��<CǼ�2B<����Y�1'=ӤE�r��<�EO=�ӓ<҆X=�r��XՆ=m��MA���=F���9�<A�H�;Ι=���*w�����#;<0;����=�s)<���<�Ā�O�7< �"=���9�9t=�j�G��<�*=j:��dw�duA=�s�<,L�m�J�=��3��6���󼍪	=:k=��	<��<�)��>c���u�<��Լ��=��w*<�i�����<av�^���su�<�\,;�½Ac�=�i��`-��ɐ��<i]���x?���=O�o�C��K�P=%ש<��!=��F��yC�҈G=u�E�<�%=v�3��M@���<�����2|<\��=9V =�S�<�mL���P���ؼr�+;}�U=��ͻ�]�~z=��x=�t=��2�����̶+�ؠ���s=�f��:�i�z<�q=�c	<Iz�f�:S0���w��t|Q=-�*�C�0=�e,=2�J��G8<�M~���D;�R8=��%<��*<a-!��:Z�f�|����<d�m<M&(�T�<��<�9<Ʃ�<0��2麛Ǖ�T�q=���:%L�s&�:���x�����=��a�;��<Z�K=(}*=�C�:P<�n�<[�:U��~�==�XX<m��0�>�|����#��ra�?f>���-=�@��>]� <��,�Q�y=��K=��Tj(<����̠�i~�=W��<O=�׀��^u=�1R�؄E<����G����I9"��-��<�3��8���<�:�:��a;y��<@Q<O�;�8H;�c <Pl޼2a=���i0=v�̽�?<3ƽZ����=NO��!N�<�U=���(���6=+J����;�=1(ؼO|���;̂��}�<��q=�@�Kۼ�(=D�=������<zʼ�t�<�q2�f��<8<��̂$=~K����<>=Ĺ��dtL��NۼK�^=��2=�x��ܾ��t�<[�<��\�iO��A&��6l�y��<�8\<+�=���<C��<͹��y<��s��,�{ �����<�����v=Ɯ��jh=`� =�t���3�@��;,`S<��?�<U	;[˺H�ۼ*!�:����P�ؔ�;��\�*��<H6=�3�^�������<Լ�_����=9��<\<�<@�><��+=��r<�#�<�h��L����d�!Jռr���= Ta<#[.���`��}�H��;�#�="헼���<�Z��\>�<�I��;:��9�D��Z[�<	��3����ɉ=z�꼻<�=�H8>��=.��H=����=	`�2o�V��r��=�O=�B=ױ��+`6=7��l�=��=Q�鼩��0��S�XqU=}/�<=��E=Ux��,������=�S!��m=��<�)�-N�=��ǽN})<a������=�C*=��O;(��<����%%���U=zϻ�cȼ�4=|+�0�	�*<�k�<��<��^<."V=x������K�l=���<��ݼ�m>�5?��#�]�R�1<�͚���P=��K���=�pV=�_�<�
f��żil��1�=Ҍ��<Ƽ���<!��_h=4"�=a:��d�<��ü� 
=��|$=��2=J;�=�������&]��S��<��'<�!5=�ܼ�GJ��%�=V+m=�B=�6��W}X���;�|=���Kd˺�����<"��9K�^<xa����Y��(�<@n�=�ؒ��l=)X���a=�7d<�"<��h���+��?���컕��<T�λ)�i�,��n����ѼC�N�5�5��I����|:�n4=X�=��&��$�;��<;��иݻ���������<�Ȼ��f=T�#=��=�S�<-�G�Οk<�D=�?��Kh��J�<������<�؆=�H�<�	<?0���">H�+>��O��g�i'�=@)�3��2>7�6�v�=���<VB�=kb�;��=<���!�=H�=r�c�?iB�����20�Y>�vڼ<�%�^:q����=�<�L�'>������𦾂�>�f���<Z�]=����@-_�\�	>�*���=`J.���1=��=�w;BT��8Z<�c�=�f,�F��I��,aB��t`=��1�Z���i-�<��]=lz�<��W��������>�@2@��v��r=��T�﬒:W�R=-���������Η�=����J��=i�">^L=��վ  2=hӇ����=�i@=�a<���z;�઼�j�=cXZ=�Q2�}�;k�gu��ҨO=sL���9�I�s��y]=uܓ=n�9<pl����8������*=ڣ<{]�<�̫���H<V�=��b�9���`�w&@�����q,=��O<�R0=�=۱<r�Q:����ע<ِ<~�»��3���i=`�W�*�;
�F���=��B�F@�<�h�<��F��<���;�Q���6�~&����ϼ%Ș�Z���.̼��<8>7:�]��J#�=�<���`�=<@@�A�P�Vs`�6k=C� �!�-=P��<@���%�F�a�!;��L����-���u���#>ku�r����P�=�p�O��<,*�<�^<d{ɼ��=Ee��='\�<�$��C�p@Ǽ>�˼_lZ��8�
|'���=`�S�R���P>1̇�P�7=����@=>�z���>��v=�=�ɬ��D&=�|��*�l=9��<��;=�ĥ;:S<=���s��<�=�k	��=���n3�z��ȧ�����r��;T��<XJ�=6��<�M�}�μ��b��v��Z���׾<iַ=�'�;8(=���������.;Z0���H��Ӽ	��	lX=���=��.�U���SD_=��j�q=Z�<O<�A=%�&��w���4�<��<8s��g�F��Ed<��3�3�<���:N�<��|���<ƈ���`�D[R��T��5�4��^;=&;=Q5h<WR�ӚH=��v�T��i���K�de�����=�]=�'E=F��<�Xؼl���G�V~?;��]=��e;���;�d�<�K���=��":��<d��<�e,��-
=s��<�ù<Q���"<
�ϻB�'<#�8����ʍr�/�ӹ��$�M=
J=�O�=�;$<�m�����i`�.| �>5E���͸B:T=4!�<�� ���d=z()�_)u�*�<�@��Ir�i�%������k0=�</M�<��"=��Z���#�/=��<��R=S�W<kǼ�r���'=(<F_`��x<ם�:	p���O���ڿ���=V/��Jd<��<��=��	<��<lQ�T�D<�#���<�vX�y�>n�H�El=9�@�V�=ܖ���T�<3YY=��n=~��8�=J/�<�m½��޼zK����~��=�q=�w���Z�3L=��!<ɿ��:��8� �[o�<��,=)� ="��=끊�%��T̽���<�ﲼu�A9�_���x�)p3����;)��=T����=lZ��as��ݿ6��o=l��=���:8<Z���<]>%��ͽ/=i~F�a�ͼ�#�<���;�S1���x=Su=���[����A��š<��>���=Vܴ<��x��dὫ�=i�<�н�{H��߼���S�����=s����:M&=����"Q��a�<��<�=��9��#=2/
;"Ze;{ժ���$��y?�������������w"<��<�����X;' =�	�<��ͻ��=���:�=�:>�����`�t=��ֽ�ck<K!Y�nμӊ�*=�==̥�h=}=��;46ͽ��<��g���?��=��;�<]=k�-��[�;h΄=�����9�=�:��:�5��r�<K=�I< �x�Y*D=�����᯽Ƣ�;g��|�ό<�{@=��<Za�;�<.�
=7_���%�2(B�-R�<yս%W�0p=����މ�=����Z<#�	�R8=R3���=�M�=�<�N����Z=�8��a�����ix<��;�%���=�m��4���N�=�b���k��y�;��uX�;��;�>w<��D=�h�=[[u�X��<�Υ�"f㽵 9��ʋ�Z�_��6M<��=P�<�<U��<�*�o ���0#<2+��kƼ�D�=��m<,�+��f<�3��N�<���Y�;�<w�؞�<�V:=N$=H~��☼a���!3߽ǘ�:'k���<k��<���=u��a�ϼ!O�=����c���ۡ|�����i��Z���ԯ=$��=M��5^[<&�)�1n�&W۽^�k�w�<���sC��h��'��������@ͼ��'�z����<ji���<��V�<�߻*��#����x2�<w��p<Qޠ�|��=��<X0:=F�(���O�px��Xᏽ>R@=� ��󺼽Y�<s\!=X�ڽ�_�;���<J'���'���f���<:��;W�B=B~����ӽ� ��[N=�4��O��w+��~��*m�c����۰:�~_�N1�L�<�*�������i?�d�];�!�׷�bM,=�+���-�>�X=�` <Z��6=S��(�"=a��;��v��F=�ü���<`��l�ϼ�Nz=�����B�<�E�=�ϼ-�N��y=�w����:��4+��
c=�Q�c�#=��T=�CA=��	�e���O1��밼\��<!�Ӽm�i<��=C}�=��0謽�F	���<ƻ)�!Ѐ=jF������X�����<e���{�V2=/e�<Rm <��"=_yc��m�<�e=,�o=~N^��,����; L�W뼷�=�@O��o��d�=��<��;�?��c@�7I�ŜZ=�b�<���%�����=)>=�4�<u&�n=u~'��:�;E9U=ۨ7<�(8=!�0�*��=9���YOǽ%L��Q���Tº7-�="L ��g=_��%������<�==6@ڼĮ=O�����S<9�<��=>y<�f�J�=�<�	��<�b����<�Y��q\�=��=A��=����ͥ�IR<t��KQ�<u�=�����!<vM�=��D=vNG;ળ�����Tv;�u��<!֔=V���
�=�	�=e�>{J�fu�(؝:qD��w=qMw��k9�*�����=�=��6�L/��?^���Z��m�<���4��<3�Ӽ�	�Q��<��;����[>ؼ��Z����<���E��V��<ᨤ<�t�=�z:�8�Y��[8\%<�!><���Ơ<��s=��=8p=�S= ,���}�,6?�t_��)�<�ѡ:�Os=Ʀ�<�	�<�Y����ig^�r��<�ӛ�h��<���7.�=����D�=��N�h���I�;gH�<&�;q�.=�\H���}^5�of�H���{�8�=�]/=�L-��.��=l��<�H�7�=��=��*����zü ��_=�)�8�����T2���=��e��p���hB�M3<�y�=�_�<��<��r<"W1����;n�5=�Sk��p���z��ى=��|��	q=��<�<r=����{�=����Q����<�`9;+["����S�
=h&�<���iк<!_���5��!V�Lt���f�<� �<C��-w�g�J���;;)5��y����Ҽ�=���;(ᓽ�\B�A��;�u���1	=p�=��="�=�Y��	Q�<�f=|�=)�e�V(Z��Ċ�oc8<"�����]=�꼣�=�2F>�3S>L j��[=e�>�{�� �6�YuG�ٲ����:;2�r����=���<w) �5KG<_�d��b\�o4=n�=����5f5="d�=�̥��=��m���;�=Fw,���]=�LӼ�)x�#�����B��/����;AJ(=��d=^�/]=�Z�<v��<��=i[7=���BML=a�c�/j���=|kݼ��=��V<=.7ɼl�u�1��=>�L=��0����<��2���I�;^���G��<m.���A�;�D�<RCg=��ּ��x<A����6=:y?��k�;��<�qŽ� ��ُ=@�� V_<�-+=<�#�(+=���<g�Խ��=_	W:�Cg<�$���,���= }����a��C8=�R����=�_�=�:�<p�ɼ�譽��l=n���>�޼��[=����,=�a<�㪼;ʳ<�;�8e=� ��7��K�O=ӂ��itr��#��+��=w�=��3�y�b<�����<۹�<X��<��^<b�3��bm;˾i���<�=�<�f(�xY�#bl;�D;a*���@9����%D������ͩ�P";Te������P <���<)�˼�<�=Ї�<��(=�|	=�u��g�<��4�R(����7=κ�;4=z����<=j� =��&>��K��J(�d�>��h��<��>=�Խ�2�=H;=��>� ��<�";�v	;=�$<�,*�2j�}.9<i��3��=�Yռ�D)�!�o=oY7���<��>���$$)�S �{��=NYI��H<���=�,���b ���,>�(��*�U=P@�<OH�<E� �5l�:����w<i�<���6�U=
�ڻh�=�4=�ͽi6=9ǥ�Xw=ߤ<+\�<�� ��9�;�c��8�ȼ�P����ܻ�b�;���<�,)��%�<�4g<�(?���t�7;>�}��'}9=�>�Y���~���=�p��v�;�	>�=kI=^<�:lS=�9��c�=���s�<+��<��9��=;��5��;<���eS�e3�=-x�<I~=��Z=�#̼I�+=i(���I�a�R=����i�3=�Ҽ�;������X�s��=�^�Ih�{�	�y�<�ou<�����'%=���<�������A;˨�<1?��C|��/=�ż9����x��!K�큝<;��{��;���<;:����;Z���Vz<x�<A׽�<<�����	
=$�=��׽�	�8Rq=�~=�p�<��<EƼ]X�9O%��|=V��<�B�G�0=�	�<ɽ�<�f��zx��}V��qټ�5%>Aj�r��c>/r�==��=�H�<-�<�R��l^<O���D;<��G=Ag;�H#�<�OG���l�."߽%�C�
�.����=f]=;^:J>M��;��<uC�M!>Ӄ�<Uk�=�s>��=�`9�|)	>O[þ�>��<�9Z;�&��*��ta����;�0����<�bI�ʝ\����<ߘX=J]��0��w)̼9��<�S{��[�J�{�e�m���ݽ�S�<l�=��=\O<Q(�u����EC<���<:'���`1N��C=��1��Ұ=�[*=٩�Ɵ=�]��V'���;W���1Y;��+<0]T�g��<u�?�Ƽ"��D����;�����=?���ި;=��<�=���<��:��;����`z=�&�=Z3K;��=�����9�=j|<g͎�Qܓ��<ɼ�?���ؒ=���=�=D=�,>;��<fAW�6 =\}p<�,�י7=~K<p5�<�DU�B�< |�<�).��<@8����Y��a)<K��:�\=L8���'�TǼ����3=�s]�Ĕ�;~F-����={Z�<�h���>�L��Y�XyL=�6=g�"�Y�$�O"��%�=�F�m�<b�:	���P<�4C=���s�
<��������h=��A<��Ƽ��<8��<�_�=�֌=�%L=�%=�F��G������i�=���<9����%=��#�R)-<��#�qƸ��o�=-�H���0<���� �<�<Y숼�\;�,��=��;&[���>#����=b���n3=��=�`�=OI=ϡ��Uai�Em=I
�������<M�<pU6����<�!�=Ye�;����2m�]T���ޔ�5�<���.]����S��=1�X����=P`����`����<����^���"<	�X���a<ܴ[��<�<�ڼ��<_��<P�����=�U��K=�|=Qz��|X�<��p���H�r�<N����͊��l�< X�:�[���j=�Yh<>�ɼf�.�W��<�s=�U�;���<��<)�_=�۪�_]��\l��ֽٞF�=��M� g������=G�����=��?=�UĽ�	��IB�<��=�ˇ<�)��hp�g���P�<��a��Ԇ<��ܼ�R{���b�̽��p=/�	�L:�;�=f�;��r\�������`Y���"���=W�ۼ�� �$f�=���=�:�^�꽟0'�x��=��D�<?k�<����=��<ݻ��#��=�r�GB����M=�S�=^�<����1���'#=�Jz�|����iu��Y�=�Q+��Mo=��<��$1���,�<��ҽ0k=�<��l�8J<�@��J�;�*��0@=�����G���;�[��ߴ�<��O�>�p==
����=bm<�=V�6��	�j*%=9F���=�Z��������49��	��Wl`:Ϟ<��H;�s��<}<����B��f�^=�9�Q���RVF=��y=�g��t؏�Ue�;VS�=�� �Q>u�A�[�<�y��)Ȁ����O����<?K=c�(�~s5=b=�<1�9wuY��'��y���;;�_�=����2�5�<�2�Ce��pc����Y=��Y����������>w}�ޖ�=@c!�����F����<A�Gy�ӿ�;;[X=q��2to=6^ʽ̏���mt=c(�ZJ���<>V�=������=�3��a�ܼ�0���u���}���,��L<Ηk�c#z=��<��<�F�:�<�/=�'��%f�<5�<�C��"B��T0����~�	=��=��5�;᡽�@��p�;�&�=>������=70�p0=E#�����:����B<�h(<U�s�E�R=�O�<o$�ʎ4��7���:={܍=-���Z߼�mJ�U�0=�==`���~��������c�p;����o�<.�^�)����pJ=(<���Ҽ�2����=U8�9� ��m<\�A����x8=�;�E��`��<夘��z �`����.=xm�=�\$���<� �=Fz\<-�>��f����ͼ`O���=ǅ�rt=+n��!Ś�=�}���<����a��BA<�s�=\�jwc=�ý�_=M>f�u���<,9���@=#��=~ֶ�t�9:R����6{�О��W�ԼB�+<B��d�=�cἸ'�<�ݺT���r�c�z�;]�S�g�X�{A�<y�<�zt=�ɽ0hG<.��z�=�<u��W;��`c�	�T�����0?=(��-�,=V^#���t=��<� ��e�<#��<�Ny<ad=V���<Ľs�H�j<���������G<���<��=a
;=ߨ��r�N��"���a=)�=�S��%.=9���&����>=�럼'V=!k=�C��Ds����<�|*�%$+=�-)��Q�D�Ӻ��<�B�8=���<y#=@�\<�W��	Y="��=�_g�@���l�D����c��j�m�#��2��������<�kɽ���<a?�.7=����~�O�5�0�%=��+=ZL�3qͻ.y<7\ڽK�<~���@�=�Ǎ�yr5���<�7=e�ѽ2C=/^���K=�����u�녗<!��;�$)=y��<�(��<eq�BG0<Q�j�����n�3={ș�O��=N5��V��L=�Yn�,>;�-Լ5{�<� ����!��J�����=		t��ݣ;�Aֽ��=��!<膇�D�<ݨ8��[X=�j�;3�ȼ�*D="�~��v���<��%��ۈ='�U=Cx2=P�s=
��ʸl<Ď ��P�<].���ŝ���|;�T$=�;���ܠ�C�@<݈9=�<x~�;a�-�!J
<l%�=�!��r��=�m\<����:�<6{������!=�O���?<*���K	=�%=��������!���K7��S��Hn��kM�=ځk�V珼�T=�Nʽl,޼cf��;�=��|<3���A[�=9"��'绕�X=<d_�x�D=� �C���x:�6=G�/��n���Oe</f���;)�s<2mG<�Z<	�
�Y�(���V��!=%�F�F>�<�I�QA���ۓ�؋��t�*�9�=)�Ż�:%=�o��ě�Ɋ�4�Z�]c�<�h[�ͣ�<�Q�<���Ve<��9�	=Ӂ�<bʘ�+�1�~���:���UĽ=�c=&���;Z���jԼ@%�/��j��=�{��!�=>��=6.�t�<��=�N���	=4�Ľ���d=�<��n� �����qx������Ϩ�<����;5K�<ۏ����=��j�a��;ƫ�y�j�����S�;�����%=�!���u5=�e�;r�ֻ=U�<Z=�<ź��mZ�=d2q���<Gu �,�üyv�=�#�{��3#@=^莽��<k��;f�򻸇|��KP=�<xE���׼��<wk0=�S<��=}^�:R,���9輞��}��̻�I�=�(软s�i�->��/<�G�<�ɪ=	Y̼tż?$	=�!��=ݮD<E}�4�8=;<��=t�j�
�=F	�۩C�]Y�<Ҹ=l,t��j;ө��Q�=�є<���<`R��a��^ ��}j<�G���GH='e��B��=E��<�����U-=�U��'=�eL�eD5=Q����_�<��<�x<�4�<�#�#7��� ��&=��<�l&=��I���<��ռ��;�琼v���C��<.��<²S:��4<�)>�5�=��<�2��������<�v�7s��%��=�CD<|�=�9<�z�;�}u�K��p��nҙ<�˂��+�=���٩4��1,�Y�v=��ͽ>�=�l;�>��cQ>?�*��4>S=����Gb��F�v�\�<wK<�%���oս*��<Y�<&7-�[7�;s@'�q	:��v<�pf��_�bwm=�I���=Ľ�=i��]A���V�=2u��ry>�Q<�K���=%h<��%��p_��&-����+$�������!<�>N�~���\Q�G.�to�;����� Z9�)�2��K='=�������QN���[���<�^:<���<��o=���=&�;�R�VV�=���;>��;�f�=�{��������!>EF��G=:Y�< ����<r��<�;X�KY�=@�v=1�� �=�%/�|��=�}���蘽�����(�����<}l�=�:�<���=)+���Ȗ=y��;�B=8��;8㶽��O=�mr=�׿��f=K�<��w�i�¼RY��q���0�;V�������I�=��y=ո�ߞ=*=d<�=H�
={~���/=ȡ<�8R<"�9���T��|�<d��~�=_�o<��ػ�u���<0y�<�Ö<m����FO��C=��Z�.���Px�=����@`��.>#E{;��<=c��<��=W��;z�<���;�}�<�N���-�=e7����=�����@<�=K��w�:�>����yR����=ˬ���=���=�<:���[��<��/��2�<�/�;Ϣ����= �i���,=<�=��]ɽp�$�<��9��v�V��=�'-�E�=�&1�vǼ���<e�2=c7�=�R�_e���@#=n�c^L=�ǈ<(�:�4g�4O��k8H�q3[=�<#=�ʛ:.y�<G�<d�����0=�*;L\�;��<�ʤ�F=��7:Bsh=��ͼڽ�E����<��#���%=,�S��\�Z�=	C�=������e�rr���a��Ш;�	*���<�Z��_OS=����<mi=k��Sb�_1�<��F��Z�=IW�=��ǽ��<�Sh�4�ռ�Y�z�=I�d�5��;�e��cʢ=}̼�H=>{�G�=sE�*��<�����ꃽ㦙=E0F=��ƽvV�=c<�^�;x=d��+��*b=��|���y�T�m��= n�;'�:ס�<:<��J=2.ʼ8�T�
S��-.<�Xǻ"�'�X̼W�0:��9����b��� ��WE�<>�ټJ�m�3-ټ�%>���=�/�=5�k�B��=��ǽˡ��珎=��=�*�=g�Ӽ�`R=a`�<�F�4]=n@B�:�Ẵ�~<@�X=k�̼�O���=�y���/�<�qP;�p���-F��J�=��<�4/��A4=N�2�U.=���<t꡽�iN<
�L=�����v=^a���.���=v���(��<Q�1��<�� @��_�=2�<M��BOZ��2��2���7=�.8��F>c��<)ZX=_cҽN��=�m;<#�;=&(�;3Ƞ�p8=�p=T�k�4c�<Z);����������=�I/�w�<2�;M �<W����d<W��;{ ��8ޒ;���=jM���w<$40��i��������<؄=[(<�ܸ0�
��~�c<�
<�i����</�=X�����%='|��� �΃�=���=<������@=>��[L�=8�<pZ����2�I=~��!����@��B���{�=Ѡ�<���<>�c�hU�=��(��h���E�<�2=~/"��E�=��j�O2�xW"���<:�<x]Ż�	=��ڽ����K�k�j=�w����|�o�<��<�Oh<��<dZ=M���?�M�����Z�.�<���<�&���:=2�h�C^�<�`�<ܻ�:$�~<Qy���=�C���I=~�%:<P�=~u$�q���D��m�=q����桻݉�=�b��̼�.߼�@>�Ł-��.=���=�wۼd��<
�=#8����Ǽ�n�|�=�ˏ�[�W<����RϼU����*=:�<|�׼*cּ��<�ý��<'�f�ft<��E��{=l9=�1�<h�J=�_d=
=�"�A=�H�<���72Ʋ;�M��ny�����q}=��,��<݋q<�dV� rx�x�μ�ug=�~��ț��б;W=�1��p<F�v=
���[������<⧝�������%�1�:�o���.�z'�=�)�=LUf��Q��K>gm��<��=��$�6�=�l��ɼF9=���F堼PK�=2/���Y=��X�!�=�J��M���5��0?��W0<��k���N�Y�C=�����N�#z����=y��<b�I�(Ur����=��U�2�&=���I�:�j۽��U=�`�� ҽ>�2<�0S=QuK��(,<�Ž�^<4�f<�>�=�A�<�s���\�I>�C����=1_��bo=?ʊ���+�}�#<����޻��2=��B==k�<ݡ�l�=z�q�U�%=����K�=���;�$�<ػ=p`�h��a0 �v�<ma<f�;�D@�Ky��Qk>�����=]��\j�=�q��+D�:������;(����#�d*3��c��p��ըڽ���;N=/�ڻ"hԽ}22=��ýq7	�{T=�u=�z���=�R=ތ��%O��3g-=�U���Ǳ�5o�p.D�!N+�nݑ:���=g��6<��<�Ǽ���=���<��<�u��䶻U����g��U��(�=����<��g��<m��;6(�8�������@0�=�5��J��߇#=R��򡀼�秼y�=w���$��X�!��=Hz�S�G=P����z>����=��ּ�+��Cw]�s�K=�ʻ��<��J��/=�����;�<K�<H�뺐�5=��;M�"���6=V2�}�U<1�d��;x�=+�H~߽�Z�=�^���I�=D ����=�A>�+FK=c_�<<��M����	>	�<�^"=�<Խ_�=�`ܼ,�1��܎��s� Y�T]�=+s��;<h�t����=/��;��<�3=�����ҽx&(>�xA��zk< ��= �<���ǼG=� ��@�;>��<�$��'C<��!�I<��t���g�����̿=J+=�*�e�.��Y=�lL�퍰���"p�<>P�Pµ���q�Ŋ�=�<��=pJ1�w'�<m?<���<������nm1�ԃ�=�S��r=� [�-��=� �ʛ�=��ڽe��=�2=[�z�
Ik��ȓ<zǲ������s�ې
=Q`ݼ%gs�p<���a�=h�q�hdO<<����8=<+�#�c<��R�L]����!Y��T�<�I�<�����<���<�w���3�9J���L=�r6�[y�<�=,0��_���^�;�I+�Ʀ=��ƽz3,�:&&>,l��,� ��V��/��=>���$;�J=� K޽�N��V8b=8}s��=A�/���<�v�����Q�7!����:���=���<WP�=!7��=��`=�,=����d�=�N=�6����<��Q{<z�]<w!�;���<���<�A�9�0н\��A�F=J�6����=������<�=G߼���<�"���J�;(=@x<��p=��D�6��<ވ�<]�!��������g�N5$=_�@��]=!�սq&��1=��v���i߂�R=2��B�<��H��	<�]5���0=c�2�<@��<�[=���<�gż���� ��<F��<uA��5�<J/�=#c�JZ0�y֨���$t�~��@��Q��<��������F3���z<�q\<g�=|�"�Uh�<�Lڽr9����'<\z=�8�`b.�̊/�X�<�j�= �=f�I�x�E=�mp���<=���'K�u�[�;��=�ۣ���=��<�FT>^�	>��U;Ҡ����(��O@=8��$=ɑ�N%�K=YK�<��a�y�ݼ����v�μ�������<fS�<r��;tZ�<$�	�*c�z0 ��s=��<�"�<�����;�Ц<� ���`M�;����W���?=C�l�>�m=����J~<,��<Oї���=H�����;ٗ�;��/���v=YQ��c�׼��=Y��<�G�����5���i<aM=���=��C�4��=z�V=���<�e7�Yg�<��-=��v��8=���qŲ=�7�=�@��L�;os<2���
��<��u=�"�9=�=�%��b����<�V��:�=Z$/��=����<�'�=���T�v��=�7�� =�n�����=���<�;�K�=�y��I`k��=��ż�aU��������d�@=&�K<�=<�������=�h%=�]�;�����B�:���;5D?���ȼM��Nݥ��a�<��1����<_V��{v%=3d;���;خ����'��-<e`<��=-�����jݝ��˸�%���o�t<[���U2;�R�=)]�<��<nF����"��#;D�=�����"�]˶���=KHl<E	�!U�����L�;�|%Ҽ�x�=�"����}�>	\���= B=�;��X6�=�,�<�q��N��=���\4��+9=i�Y�F�g�`ʻH���3���Gj<�r=��	==�:=h����D�=m -�*�ݼ����M��Z:%=��f�������=1�;�+=9,�<+Z��`+=���:[�^�~�=P��穜���=^��o@�<_~��g�1=�e�<z��<��E=s�;Ү<b�)=槝<���>�`��M:&y�< ��;0��΢���M=(Ǜ<W��<e����o=7!�1A���-=8����<��>��;S��=�k="���bn=�U=VN�;���=^R+_��|=ͭj�P0=�6	�/�>�I	�5�Q��}�=%L==�㣼�?�=�l[�
wX��%	�3��=�tn=-�O<��=m1�<3G�����=OV�m�p�����A9�A�<�Z��"� �K><�=��R<�r=6���q�s���\;dἎ��<ұ@<`����N�Kp=�D�<�,�<i��:�4�:oSI<9����S��<���A�9��ɼl�s��;� �aF��O�<6���
4*=E��=V���]�=�<�J��I���^�<���A�����#�=�u���,=��޼����|��6�=��Z�P����=g�μQ=X=�e�Yt�=C$U=��W�C`=�]�<�̡��s�=M�`tQ���;=�H�	9���]�<.�N���h9S=e�f<��q=�놽�����-=te��ϕN�h�f�VS&<�c���!�<�x�=���;O����/=B�f=
�:�	��=H@@���!���y=�O�<AW����<C�źd'N�OZ=�k��+���q`<*G=�mJ=��!�����'�?�Z^��;I7=� ����g�
͞=Z��=�M�jJ�y��8�<+��P*=ۙ�<�1Ӽd�=���9�<��;:�2��=>���B{��4=�ᔼ�μ/�$�|������#����	������D<z�^=��y��"�<�P�=i���=An�P�=�����?�<f�=v�!=.X����X=�<�xǽ��d�T_m���/;�0=�r����	���=���=�V=�/��[�. �;�w߻ p�;�"=��]���<hOJ�{\�;h�����D��>�H|>� $-���0�6 Y<�p�<r����i�J����N�_��<؆�<2�ս^�D�D&���!*=��=��o��BW�#��/�(��܆�ꚛ�x��԰N=7�&<e��=T�h=IZ[;�����<�Z��'=�a<Oċ=�G������<�<���=�_�t
�����=�o����1���V=Z���R]��)�%=5P���S�<-� =F����8s<y_=�J�<�:=�����4��W��=�Y=����B�=L4���%����;t�ӹ�e �Q�\���<�n�;��߼�5�=^&�^b��I�5=���T�½�X����=��X��I��QE��~��5��q/�g�e=.ȓ��]8�=N�<�O��?<Y} ����C�C�
��H�<�C��*(�Wd=�	=9�D���x=L�5�a=�>V�`���$�_���D��ߚ=���c���s��$G��ݢ��w=�'n=й�<˒�l��<v-��*<�=���l��_� <��=G޽����9=e���P��)�<��?ɮ�&�;n�<T���Pa?�#�=2�|��P�l�;�#���>���&R+=�,�
�}��<<	�9�7"=��,��|,=�2<�6A=�x�:d䬼Y��<	�N<�N�;I+����2��t<�9<Gr=
J�;n�黥�=������<_�=�x켇��C���ΐ���������Ny ��û���ɺ����H��\������3Ű=t7[�N缽��=;�g���e�-}�<��{=$���A=t��#���[�$�=�wü);~�=�5X�J-����<�W�;��rB���p=�Nu�'�l���L<{?=���V<v!=ZA����,���=t��<��C��;=2��hY=�I�첦:�ƽ�崼��]=ee�����e��o+���AŽ��=�2�<o@G�5��=f8���C��a<k˽c����:���L=�U`�S�W�߄���r�=�����x�<���)��<�B��l��=��Y��]��������"=ׁ�<xT�<Ւ��#9 �]�߼�v=V�<��*����J=n�<�>9��E;\�8�ȼ��F���~=_�_�Y����+Š�O$�����=�����n[�j`��=?Hڼé�~A�<S' ���l<��<H�F�Q���ڼ��=���ʗ��g�;� ���6��ܿO=OH�<�<
�4�F=9�=��<i�<�s���~���=o�2e=	1�;d𸼳��<�U��}����������1�=IL@:�=K��c^'��J�<�ǽ*��d���.`=�p;d;u=�����-����z���q<M:����a��^�=v>s�O���H=/��������нzD���������ٳ=p����׀�U6��΅�<����M�<�xp<φ�������;&���
F<M�<�HΒ�Lˇ���b���Q=����k�< �!9gc��Pյ<�Bf�aZ�;�c`������.�)o���K��<���=-����̋=w��<>���!=)���X����9�T�=���|�x�n��Z[�N�ǽ�Ƚ�f="��m�z���)�,=�;3���<f5��J9]��P-���Q=ν�Y"� B���1>h\R��"�K:��U[=-���y�=Fy��ܘ�<	�n�<��O���r�R�>���=b9�;���R���|	�Ɯ�,�=�_\���;7��w��< ��^��<�н}���s.�"��=���;*>p<~��;bxI�r+��٪6=�Q���U����
��?=[؍:W�Q=-=��t�;4(��Ϟ=8_���G����>;r*�����x7 ���|�e�*�߂�=i;�*	��vĻ6D����=�BH��!P<���;+u�<e�５��T�oT
�;�=a��<5=����R��0�<L��;�U���qK��1>,v��H�<QF���q=x'��)e=%�}��xK�|)`<¤K���<Yxν\���<���=%�m=ȓ=��?>̉|��%Y<ɬ�<�#��:��㪙������Fͽ�������<��=`����t=�RW���H<�,��3K�<8��<�?^�ԫ�<�4���<�̍;�J��=*�T!
=)�j�ٿż����6=�*�<�@f�iOz<�A�Ξ/<�v�j}��2���~��c���?�=�쥽�cz<$P7�C|�L+��#�=��"�{)2�}�L��>w#������m�m�W�{;��;,���D�i_!��7�:
�=�ݽ�����]<�Ӕ<jm漤��<~G�Ȣ��(�<Tڭ<{�<\J=���<��(=�t�9��}�ֽ�1��=���4>*8��ы=����`t=�=��ֻ2o�YZ���J���3>zR6��+K=0�������nf<tüg����C����=�~��u�޻i�n�c��}����<��B�t3T�8�]�xV�=
���˅=��,�A�_=�[<2<;�<�HI<I���<p�����;�\�)��<���;zb=�+��<��}�*j;��@q�0ݏ<;��<Ӭۼ�S5�*W<zE�R@����)�'�=�����#<ű�<��=�0<��,=!L��?���9��}�w=�F�@e��C��R!�<~�ɼ3s�;��r.���I�[;[p*=�ݽOm0�)̿=��/��H�=���d�ҽ���,>"�޽�4�=/
���|߼L���y���?='��;um�=~���pK׼ě�<��e���=�E�-��pќ���W�B'=��;F#p�g^U=�a�I�=��K�u^=^�彴�A���2�x�">����ރ��C<��m�=��+<叹<����s�������=זh�[�<�����C=�E�w==�.�؁��&�� <=�p���A׺􍣼�<LJd<�=8�����G=`z;�fʼտM=�
��؋�(A��c+;��=u���_�`��i�<���="<����t=u6��k���e�?<pC����d�>�����p=qF�=�9V�R��=����k��D��=�O����쏽X�����=������=�2���w�;��=J���A�5�Z�;��=�믽�kE�/�	�O/�;�V =��#�-`ӻQ�}<]��3B= <C5<F�<�_ϼ���<[}��t��T
��ی��;��V��q���U̼4p��m�<C����+��m����<k��<�E��t�;-����N=�p󼟦�=J�$�`����ѽ�=�˼����E����ؒ=�d���J=�J���m��о<�i�N����Q��G����=��.����=�.���cѽ���8v�=�+˼!.�=(�����=3�l<��ѽ���0�;�&�</�ػN/=<e���j��@��H��P�!<42мlYW<��=�Tw�p1=�-�<@mK;X�?�M糼dC�G0�����K^�Xb�=n�eǸ=���F'�;]�<3<-TK������޼�n=�5#����<�]���+|;(�<Ȃ:�$�J�*������x�]�m����
S����<6=?�=�Dy;�%��4LɽK]��5�*���Ǽ�b����<t����8=���c�߽97=���=Uo����=+5��	�＜�=��ܼ'%��+���N��=���=-����=����e� =�x����������=��=U���Q��=0��:�2��ԁ=j�ý.�%�6*J�-���N�;aĻ�d<��Ͻ@�L=;����=^�=
=w&�<�Q�<*�!=Rz�<�R'=�����3��*I�F@,���9�c�6�H�8<��˼G1<X;$�;��=��,=Y��yg��bǽ��ϼ��@�g�������5=�G'<��=Ҟ/��%�N�>�j��:]�8�F�<��Ǻ)<��<���=�DR��s�#pr�<�k=Z�G;%���,~=dB= 3�����=SŲ�����r*=��=	`��l�>�AZ�����=���,G���5��V�� =X�a���ؼ��<E�={߹;ܺP<)8�<�C�<��=�� =����3ǀ<r#��P��.=���$��ޡR�A=!=N.�<�@����=�{v�f�{w�:l6<�i�N]������m[��2ܼ��<m����<sd�;GO=0��G�y�l+�bd�T���u���׽�=�4��~�<���<�:���B<GQǼѼ��v[%=��5���=6F-��4�����11�V
r����WoҼ=��=?9|��'�<��_<inE<����|8��2���?v����<5�=S ����=C�<����5����v���<I�=Pͭ��=,>�,��p�;@��=�`��x��W,���ν:��n�̽�H�G;����=_/�<Y�&=$f_;&�S;hQ=x<�r��)"����;�Y�;��ؼV�<~��<F�=tC�< s*�;L�M���ե�;G�d<ٺ�� �=����n���yO����������<�v��lf]<p��G8>��6���1��O������h�S�<
���=5�==�/=�qڽE`���ަ=Ъp�ko=Z��;�\����t��=�[,<���������=�=u�O�L�=���t�=����F�p<�nf�`KS=��ۼ�|��=>:������;_c<�f4<�p��5�<�,=Y��{/<�_��&�,�����W�����潉�5<�9%�9p���<=�u������=@��<��u�O�!���Eg�;�◽�[�=Vy��b==6]�=�v��1�]������e��g���=�ܼ���=E_��q��;��[;B��<�5@<�dD���=?}i=H���W��-���w���׽�4˽Q�콂�}�}M����=�Q���E=��<��<����a:꼕K���!�����r\�=V�; S����� ���6��R/�1W���v��'&Ƽ��=������<S�޻�w�;�=�;�ؽx}����<n����<�b&�I�5=�Gt�uy�=-�=�7=�t=	�<t>)��"Z��^(=6a�;1�Z<=oɻ;G,�u�<LI=�:���Q�<S���g<�P�U����q�=i6���?��&*�<��> @��u;�`c����:�����}���	T�6��<�A*<���"
Y=�����q�=�������Y�ڻ;�A<A�C� e�=�,׽��ֻ�=^�{=b���ݽ[����e`�J�z;K�<�P�X�{;���<�!�=#���Ν><���<f9(=\�k��O=6��� \���&��S*<t�<h(����(= <<H1e��F=K���kA!��ͨ�16�w������v��\�;�ϧ�m�^=]�8gM��T\<�n~=���������� �a��<��?=�� �a�:������=<�j�0�i�fj���T!=�;#�S��墼��2�KL��5,>�ξ�orֻ�x����<ɖ�<I�G�����Y����<6�E=hB��WǼ[N)������l���D=�K��x�=�[սf�U=�KO�v8.=a�;�Aؼ/E�.{=�l�<�9���W�~~�<��r���9�j��˂�nn��E���/X�D����
�� 6�=�<����=$S�$Ϣ�>+�����<?g���*���=��,���ټ
�'�s���tĕ��J��Q�<pb(�g��>#W������\��<e*=��<�=ߵ�q"(=P���!���8����8��^x+=���;��輚��<9*��m����=~��<��]�?�r�,y<�nC<	�̽�S	<�������:��߽�B�B?T�R
 �auL=�q=�3��<�ҽ,IͺnϮ�|�A� �����r��bm��bc=�|��S=@Lݽ�G=+O=�X���S߼,H���=���=4��;}�<��;[S(���������G=�h��S7<��=��l��`�]��<�G4=눽��.��m���%нb���pH=�>��{~�5�������F�:��;2=���VM;��'��;� �g���ý���@0�=��˽�eM�7�h���<�����t�z�ͽ�#����H��=�d����@�������F�!6���s��H=a����E������.����+�or=U��W.��/���&E:�x�D>�i��q�l��q�Om�=>M����ս�\v�������[��>�=�Ѝ�k0;��k���p���i��R���:&��<�X{�O��=~���B|��:�(�I=N�ѽ1)8�q�\�?lc�@�����>%���|ڼ�ȭ<�x�5c���<ʯ�;��3<"�!=$�;��� %������=�ɸ����%3�<��<Q���w=�ǥ�����V����$�ּ)�9��q�
�I��/ʼ��=�,��#� =���.�<\ 꼤0߼���2��}�.�=�O<��=�i�=@��<P+`�YTG=���<��<mH��]��hiY�7�x��z�PTJ�ӎ#�yW��޶<����w��h>y�z��<�߲��)=��7�cQQ<����#�-��;�zf�ؓ�<!�#�F�c=�>=����a%
;	nQ��a��=��<�����;�;�e���< ~�2�۽BV伏����9���=t���Z�p&�t��<��������\6=� �ச=����൉�X�}���<�ݽz�Z�ؒ��=�WB��<�<�ż?=*<��o;,�<Pg����6<[b����u�XGn��Ǜ<�3�)�ݽ:5�+��<�>N�� ��(���1=R\1��'�=��=�������,�=i�q��=��aJ�w�s��^���=��|����Xq���v< ���N�����=�I<��B=A-��H�x!����<�|�x`�� b��P�=��;s=���=m���6=\�3�!=���!=�:�:Ի;�\�<�==�S뼿�dh2�ѯ?=x2�	�����������M�`z=�B�Y�<}���;�;���� e;|�߽Su���]��rb<�
�Q�%��+6�E� =NЯ���=�\?=�)ü�Ez�V�J�3u=�Ψ� wۼc�(��٬��>̾C�O����e�[=����?J��l{���=r^;��s=�Bk��<�;=_���5H�<D�E=u���|��F,=4��<��@�U*U���C�ad��`�(����=���/@#=B�=��1=p'���;��)�I��w�<����c���Bٞ<s[�<imn=�޹�n���m�=��u=?�o;�M���얈��/s���=n�ؽ���Mk����<VX��~=�d���ܼE#�=,\v��)���h���'=#�"�� ��:gm���^�R^��ʠ<-�������j �=�=0�j�6���t=��a�H=����ɽst��o�<od��@~=d���\I�<�.<��mx<[�j��ў���ؽ6o�=9$��1��i���_�=�+=�2���Aǽ���LyX�+=����,H=괽���;`V����=������;�B<U1<�����_gW���|<��:���wH�נF���$<VG=�=e\	�z�<���<�z�<��#<�T�����t�����NP�+S�=;<&��.=�������=��Խ��A�]�#����<i��r���x�n����Ћ��J�=U5��q��W�ٽ�ݽ����A��\�<Pr==c�|�(C$=����=)ͽ���=f���ZP��S��А�;���;�d�<͜;���42����߻(q ��;�S��I�<�+=¥�;�+�Z��;��u=Y����쀼�G=�B<t=�p輰�ջ���EU.��l0�qn=���M=<]�w�Ӂ�=`L����<��.�D���gU�͜�;":��ؼ�yWk��u=�
���y�=�b!�"-���`0��Vz�� �[����Ž��=m����=�q�;n������	N=*F�����<�b�r](�����t6-�����\����=G��Y=��-�U��=��ҽf��:~�fN�����@�<QĽՆ<Q\?�{�м����0/s=�j�v�g:��R=�)�7U-��V|�'m='5���D�<qP�fw���a��U�������<�Z��%�=ð��9�@=��N���<�8�N�����;3!�פH���<������8�һG��[�:�NM<�@�;6�T$G<<�)��i�����j�#��^<�p�r���ܺ~k=�w�r=��'���۽��\����������I�;�汽����+��1Ѵ=v�{��&���V ����=�{ȼ[�"�,�,�C�������=�.���^�]�罳��=.�7��5��î��x�9���n=��g�.Ա<�n[���=��#�t��<�D�y�=Us��U���=`<nǼc�=��;�����fzR<�=�'�<{ʣ�2��j`��g��!<�=W��'�<Z��V�w<�	���Ņ=�=�������R�GZԼ=��	����ҽJ�=P�z�='�#�$���#�����:�R���ཱི ��Нp=��q��i�=8���8�d�´;L?��.޼��.;�%�^�"<y`>��;��#�N��5N�����tϽ�SH����Α�=�"��=��ý�ؽ�����+U�K��j�=�Qk�ԉ=�d�_w&<_��tr��z,����<h4*�����Q`��&�=�Ž?=�������E��Y�h��UR����{G�����=)�ཛྷ[>Cv�<=| =h[����;�
Ի ��1\d�,�N=��+�ؔ\�Nh������?�{�ͼ�P=v�<�%�km��}�<�=1*m��r=l���A���?]�Uμ�Gt<i�=�n���a�=�%��[��j*��F��2���켡Q�h]V=��2��=R�Ͻ�҈�Ӎ�5YX�uz����*���u�)�ɼ��J>�{��Գz�T��FU=�Y!�b�<�S�����=�a��Ae=+a;F�y<��<��y��<��<)e��_==�P/��.H=�$𼖛_��=�=��=�2ܺD��=;�	�� M����ã���3�?���r��]J�<�aĽ(ż�s���=�\ٽ��H=i۽�[\��%�
���Ƚ�׽�ͽ���=��D<�e=fY���3{< �Q�[�O���Ž���!弦�=��� �7=/f�<� ʼW-��3/�<{��3Z7=k��I=k�<+�d<�F���������eko���%��M��O�p�gҦ=��o��h= ��ok=�!ڻwv̼7L�6�_=y)�� �=6b>��o���Y̽����f;��x���d�����;�g�<�P��a�=08��I��,��Ň�<�������2O��F�=�*�
B�;K�˼Ts2�0�l<����H��/�żۏ,�T����0�`�r�O2<�=$=	x����?o��[�8;��k�=�_i<q��<K��T�<�6/<1�=�P�<9=�~�ia<�X&����<�D߽��Q��ƕ�;7 ��������G�=�m��`�<ܴ������P�=��g3�k��<䴽�V<峻p���t�=�X>�淽	z�����Dn�� b:�E�K�Y�]=rX1���=��C�A��<�PX=>ob<<�<�G�����<�u
�~ູ�JG�s������.a=�k��.x��3�=/�ü��-�;�!��7�e﷽���ն���<�5�8v+�[\��
��=���<@Kq�1���]�l���2I�P��^z˼��T���=o
�;d%�=�e��``����b=N����ݼ�UD�1G=���޻9<)K���̷<�W�<N=�Eк�	���D�<�,��b�=�v�<�=P�D�X<u�<�I���T(�:^F����E���&��:�53��U=E=�`"=S�:U0�<�ʞ����һ<}=Q����><%��<:��;sE�D����������]<�W�<6̲<y�<{ֻ�;h�o!�;z��<#���`=����-U��4ռ���<M��S=T�]<	�g<���<��-�lV<�Ӧ<�P��`����2=�,�5�Ƽ��.=��Z$=@�:��w�XF�'��<$���1��<p���j�����<���;B;ǥC�!�z;N�;�u��t<�� <�e=�}��,\<�����<�������Ļ,�����㽜V���ۼOf=��/=��u�;2�=�?.<<>��¼;B��'ڼ2X<���;6
�<D8.��ߖ��פ<3񀼁�>=e�=�@�f�=�wQ��ԙ��42<�z���+�:������J=k�B)m�]�Q=c���&=]��<n��<��#��dr=�<�@.��컲M;9�;L�x=��<jO�:d����}=g�4�\�=6N���v:1jy��s��� =g�#=�s�IQ���:��������۪�h�=ng���4<�n<k]!���˻����L�-��z���)��i��8�1��k��pmx<7 =�]=�x+���r�ۄ�;��,;w����ȑ�eh�<�)���i�<�nP<����똻@�ӻ�ڼUy<��?<���j5e�� ���(=Ͻ�WFo�f��>t�;���;��=�bQ=]z��A3<�VȻ�f<��h�1���9=��m0i��=��<���^�¼���;�����2�WWĻl#u</9<�t<�R��zJ��nѝ<{#=V~꼔�J<X�|<ʔn�^��̼��$<�E�;��������L�K=�W=߈;��S�����:���<V�><yc��� S=y�+=��y�M�H��ZT���zǼ�<G3=J9=Fj����ܺ
~=�����bۼp{�;L���R�<�I���7&�n�=���B7�<��2=9/ �7�:8<Ϲ���<O�$<;!Ǽ��ռK9��Z�<ѝ��<�9t���"���B<(������=�(<��x��#�<����ub���&<�:a9ʼx�廈^-=�h��&<Z�	;2A��������|=>'=��E;��ʽ7~�8�<��<?=��OPH=�<Ӽ~��#3����AY5��%@<���<��f���x��<1��<�ٛ<�Y8���<����f;�(f�9��68=+3=.��{��</%�����@ʽ�;�=1�
�����L==������ܼ۽<��K�,sm�_4=��<��/����;b�8=K�)���)���u"�U=��֛��	޻���<xS=كx��y���E��RM�_K����p�q��<��V��=6�<U���7�<B{�<b5��	=ڈg�n�3�i�6=Tv�;���n����:��)=��<[1=����:A$=��66=n���.;W:�ӌ=�Ҡ��
�<m���a#=೽�� >S����,#=GH��98<�oӽ�=�=��?���?�� �=[�o=��7=PB=}@��:r�n��<+��;`�<�a��E���Sy9�4��<b����;�pK���,�5}��d�*=���I<�	��/u<t���v�<��\��<�%�<�F=�3=Lh�<�<�<Ƕ�;��<�D=�[�S�)=��@�0Y��[�<S�=M�f�҅J=DI4�%XO�Fb½�a1=�;��9':���/�<n߻�n�̋��~\J�~�'�?�%��KԼ���=�e伥x4=^B��	Ƚ�VW�D���O[�]����V�/l=@�X_�=�Ҡ<��<��3������Y��=\R�<��;���<�R=q'S�Kݸu&�u�Ľ� �]Aʼ��I���=C;�� �&<w�D��H�<h �)���J�<����0�=���n8Ǽ��w��Uo���4��~��l+���(��!t�L�K=xv��!�~�A�V��S��?^��ǐ�C�\�Z����"��̈́=;J��׳<��-�*x����=�T<�<�s=��=C����|L�V@g� '.=\ �#;s5������C�m�0=���<����<�U�cُ�@=4�p���.@����=q��9�@?�4n�<؇ʼI��<�(<�_�� �2��k���-��[�"=�2����d=3,=�PӼ$��=���ϴ���=�&ȼ��#�r_��Ac�={K;����G��<ϲ�}潶:I�h�޼&4�=�Y�EB�<�ȣ��kB�f	<�?�<�iH�<=�4�<�!Ҽ���<�q��ZM�<����\u�<e%�;��m��Z=ZI<Ic�;l65=TEO<��x����hN�L�<�"�½��������+c=)����l=FC'��W]�J3��{��;}ǽ�#.��=)�"���=�|���Ƶ��Zk��]Ͻ�@%�ig��g�����=�^���G=���<�A}<'3�:��<�� ��]�<�&�,U���ּ@K>�u���b��4�;�������pV���0��=��Q�wY�<��<_<t޻|(���ݢ�~Ü�)�	�O��=�ܜ�Lm=�m�[�*�5TK��[�'��]� �;/��=	��5��<���~¼�|��Ս"�Hf�^�"�[8��=R�+�b2*<�Ů<��=7,?�1P �)�l��N�<pl���"�����<�br<����+�A�<7�<�n9<����8���;g<�fI�q"=�e����<���/Խ4�)��a���nQ���4;��;��B=y�	��r�<N%���������c���:¼��=}���#N=��:���T=v-��E5�H)�<8�����h��=^�7�qR�1����ʽ/d����3>2<~&�=����8<������Q�z�g=s��;p�V��<m�Ƽh5/��<������;����\�=,V�<)~"�	�H<�4��ܡ�V�=��M������ټ9��ϸ��dڽ�^����?���G=����y#Y�J/5�E�=��L�������(��{ϋ��J�=;t����=�5��7��� ��΂��U��D>���;a�=�]�s=���<ܺY<^�Ÿ���t��++�����X���ű<��<�H��R�W(��Y�`�i$��=ҽ
��<Ux�=QP	�E��<�YF�cu���r��Z����w��nʮ�1��m�7=�Ų�}b:��.��q�l��׭�O X�|� �N_�%�f=*t����<8�Ƚ�>�<nqV�0Ƚ�ؽ�;���a<9�=#����'<�'8<��n<�;����7;f�[�[x,=Srn<\�f<��<{,=��+��r���)�僽<�9�����<>�'=�;�;��<��;<�T��������hI۽��27ѽ8��<ak>�H����<껼%�/=&]��pս��V���3e�$��=O���u��;���[ܻ�E�=\4��� ,�Ծ��m�ļ�M��&�k�7�=�	���I<>)k<�杽�iq��&���h�<Ej=�_��{Z=D��f�9�#4=�'�<.�o<��8< �,��a��'|ܼ�=���;�@��YU|<i�Q=�����"m=l�y<KD�VF��D[{��2���m�;�+��˅��/گ�7ľ��c=�=cܼmD��2<
:StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp�
+StatefulPartitionedCall/mnist/output/MatMulMatMul2StatefulPartitionedCall/mnist/fc9/Reshape:output:0CStatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp:output:0*
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
*(����s��c��<���; ��VN=�5����;��;��(�2=
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
,StatefulPartitionedCall/mnist/output/Softmax�a
EStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:  *
dtype0*�`
value�`B�`  *�`~�S�ľ�c��;=���=�Hǽz4r=ac��\>�=c.��9;%��<gԎ��$�9(����֝<ن��G��lֻ��>��=XÞ�K��:!>T[̻��=��f;ez��5�	����3�K�D����|�C0
�=�Y<2���H��<���=�n#�4�:=�E�<���V��[�����<�.�����9W���V���P;�b�=����+�@=u\���5$="�����<���;�,���|�<_U�9�����<��,=ۧ�����<���<X�!<>�9�����P>��=$+���d$>Y��<F��;��C<ZgF�@�����=Ċ/�zwN=l��=*<�W�v�r=�5��_��=�<�='�<�t�=�9�&�<�q�( ���`�k�=U�=�zU>o�<L>I��=�l>Y�=�ZZ<H�=E�y<y<=��=8���*=�[�=.z�=t�^>`�0>�
<;�Ǽ��J=:g����ͼI|a=@��2��<y?Z�E$=4 <�>6�<Kn.=?�=���<�v�=��>*z>ELU=Ha>���<��V<V+�=hؿ;�L�=Q=�Y�=���8�=TR�=!դ=h��=�� >7d9��='}�=�4�=�=�,��Pᔽ�:K=b͞=<Ƿ=�y%=V����Ԟ���>˞!>�_�K+!>�$4�?=>=~��[~���ݐ=Q!~=ﺼ'Î=(�aNԼ� �=j���nk<C�>��=c@��&>�μ�V���;��7��M�]ؾ���Φ ������":W�=���<���a�Ľ�=r���4"%=}s�<`h�<�뗽��=A&=�g�<N�E��t�<{���ټ}�o=:\���M=�X=�!�<?w�<��Ž�; ��w�=�W�;���<땀=�P��OX=�g༠g���B"=0/��,e=�=4*�=��Ƽ��;�!<�t�<���=�)�=�3���g�ׇ�=�d4=f=� ��+F�=K����߻�����x��ν���.E=�J����<�@�YE��X́���/���=� >,<�J�=��!��=����5W=D+>Xд� ~;���=|�J�\	��m���x��	#�.B>M�P=y��=[�=ɠ���p�<��
>�ҏ<�2��o�;����~�"m	����<���=����=$�=�7�=���;JMh>�?=�iV��ٟ<y�=��ν_�=��"=��Q=`�=���:JZ=�I�=0N >.�;�*=@ڭ=��F�}��=����k>�)�z�\�(>;�Z=T+����>�=\7�<��T>��0> >��=�>#>�	�=�p�p{>�S�xuC��B>��->��<�<>N,=�C[>Q�>�$>/�=��%>��=z�n=+�>�ZQ�\�">���=�z=�4�=���0Eh��1�.�=�m0>�,2<���=�ۆ���>����ս:�=GW%��>*<k��=�`񽍆���ԅ<�A��&7��Q>D��=��=K�=���<�W�=��=*�I��=��E�0�[:C�5�a�G�=W^�=]M7<[�
�x	O�q?8=�f�=�]�=H&���{�G����4��I=/ v=Ɉ�<H�u��6�0kM=F =����:�X��/��x9 =�o�<��=p��<7y�b��=��<W��;/�
�t�_�=C��=��2=&��'OL=A�l�rFA=Ǯ�������=������h<{Ly��^���I>��˽���2�
>���=�e�})����G=����A�����A)����=��� �=��;a��=��k=����!j��+�=� >\�9=mD�=}�e��Ɔ>��>�4<s�=Pv,;����?\>� ���̉=�@�=$Rs��\�	==��:>�i��>���<�2�>̏>#;?�<9%A<'���4;��:�o.�����L^o={
>?$'��^=N�=���]>P��<�����kp�m�ռ�Q����=�`�^���l�=U]o�(�=Ƀ�=��=��m<���<��ü~#�=�p=�#��򺾛�9�g��D =	�>Z�l=7E�����;�<��H=��=��<�b>�˨���-^=�!��C�=|��RT]<%�l��>���<(�ý_��<�I��"}<�%�=���'�=��^�"�%=M��=�S���dü�սm5�=�4�>��;��:��Q>�}�>�~�>]g=�#�=f'�>�[=�-�=�V<W�q>�|�=�`��c@�;=E�����>:��>��R>��6=��=@��=��>��>yI�ֳ ?ȑ�>ꎜ��8�=<P"�W%+��s ��5�����>M�[>�N4= Cʼu��>Ҡh> B >��K=��>����<��s=�ӽ0)=WnS>�j�� �>��>Ƅ�>����O�=��/��Vj>�>T΅=���<��.��	��s���r;�Љ=��<�N<=l�	<xF:��O�c�^;�&ս`�==Dݼ�1���:<v�˽yI)=��e=id��W>�=ۊa��=<z�\;�/=�.��ܙ�=~����;���;�-=�Ƨ�^����k=���=�=sl>-��=��=9� ��0>�H����=����d�B>yN���}�;�̦���o>�;���=ҕ��B�>N=Y'�=�����,;=�c;��=��=��>8�=��!<D֍>���=�վ=D[�=�0���T��-=��1:#��5<�R�#�<[R<;�=����g˽�|����
���<(x��~=��һ���<���FF2=A��<�� <�;����~���_;}㮼���<MJd<ЅW��ɰ=#��=qҔ<}+>���<� �=2]>K�T>�>\��M>��=��=���=��ͽX�=O;�=}��}��=��<��黙3j>�n>�I�=k�`=�N�<���=Z6�=��>xPm�*�>5��=��6q=�j;���R{߽���=u�=XF�q�%>�OA�둞=c,�=,�=u}�=��B={��*_Ӻ
:�|�Ƚ��+>�潽^ =�p
>X��=�Ž�[�=�m�<�-;O��=?r�=7 �ЛH�h䷻��ûվ�
ٻ� �
 =�I)>�=B�P=+�?�x��>kG>d��<&�P���D��� �=�����导�&�=[; =J�&>J�<ZP�>泮�n�>�*-=vm>�Fc<����{<�O!=�y���ƽ��?=_Di�Cｅ��=�v3>ux���=�\-�ki>�݇<���=]�!>��������=�$���!8�-	k=�0,�hy�=��>Yr���sY�N�b=;�=���=�5�=���<E�=�v��MTǽ��սW�����<���ș��{T�ݰ���C= �<�\������=�U!�
r)=/������`�<��=@A��S��[m����=5�A=��=��<�^�<�^���@=Z��;�M=���<�K=�f;k�<[Er�ot]�-�<L��=`��=%=D=^ҽ5�q>��<aq;���*��Z�=\���b��=�5�$�(<M��=����W��<"�x=ڈ�=8L���=���=Y|���=����S��Q���-��^��*ӽ@�����׾�a�<2:>n =�;>ؚ����h>l	C>�O���{�=w`N�lW&��8�=k*W�(�N�>ؘ���y=Uw>v,>���<�:x>��<��d>
N�= ��;�*y=� _�Ob��M����?���׽O�U�NC�<{n>�e=�8,=�Z��0��=���=���s�>w�<6�R��=��G��l�=F+�=�"���jU���@>�/s=�	^�]:�=�a�<c4=="�=A8=�5w<P��X#�y��-�A=߹��5t��ա=�b���0r=�X�s���q<� =Oa<��񼠽a����Ie�=����=<?���@�����`�"��Ђ�TE�<�w=x���I�2�_��C�=]>�����g��il��;b��A�i
���=��=n즽��?�w���x�=#�%=����I�=�	<1�ӻt�=�*��P�<3�=y�M��_<-O=���<�3���=u-=�G�"9�=:��?]$�x�<��)���[���.;�C�=k�� �>� 2=L��\m>�D5�\�>�A<��0=��=�G��v�=r��=�>�=`j����=��r�\�>eӭ=���=n�=w��= 0�<֡�=���= ���i"<M�<R�<�咼��<�6-=��s;�6�����|X�<�t�,E�=�e��;0�u��-<X���ԇ �Ls�<s�!=�-�<�0������>��ϙ=�q�v��<Gzؼ �׽��ֽ��C;c�c���<H<',8�����<�n�=�W#;=*�=��=��ν��=
[J�m⯽r�7=��.�x��=��弫�=f�=&�~=��<;�����<i����=�=�<�ﯼ�v�=n�<��="k�=��<�_�=�x=�X�=�Tv=�r�<��=�"=�\�=:�<��%>-�>�>8dk���>U~���Ȇ<�F6=?��<M�=�t=nǄ�,O���;=>��=�B>6�Ƚ�7_=�I=�Ѱ:s�>Q�
>c�=~0S>��;>�R8=a�>u�P�<��>m��<���>�]>[{�;���<L��=��v;�D=��5=�؈=*Ґ=��==��;��D<���=��=5�*=|��=ICԽ�$�=G��=��:<}��=�1|<	�.>�.*>��>�>p�Ǻg=yP�#�ɼ�Q3>�}W>��=f^�=h�>�K>+!Z���f=s�e=�ڪ��P�=��<���<�9>�;K���>��=<��=D�ͼ(��=�S=K�>0��=Od6=9��=�~���/�<���W!�<���̻{�<��(=���;7���+�E<�m�<��G�X�^=4ѷ<�鎼Br�+���ͨ��<��8��֑<���s�x���������@<�Љ;��f<?˽��؞����;�_S=�)=_<����nH���->,w=�:�=I�<﷣�:!�<N3�<n��@�V�7>F=�rV=%��<��|=M&�=�Ii=�J�Ѓ��Z�=g�=��<�=�?��=�uμB�W�fu="<�<}�	����<p��;B�A�2��s�<Θ$=�'�=]�=Y�����>m6�<r8M=��=�D����=TՕ<I�o�8��=��i�3����f�=1�=|=���=pRR�ck8���=�:�;=�=�Y�=����7D�=���=f��;̬d=q�=9G��<�=�=W�>��=�!�=@(=�=�p����<�G=W~ϼB�=N�$=�Ut�&j-����<�B�=�n|=�>�#>�绥��=���=w�<F�p=0�Ƚ�>3o�=B`�<�%,>��=\9�=���,�=n���;���	>��R;������>w��<���=�b�)
�=3��=O�=<�IW�$�j�K��=�
=s�5>R�����A=Kx5=m:�=��X=�dE>D��<�>�# >:��=ŨB=��켴N8=��=��=)��=��<Y�>k�=r�>P��=[�}���(>ꝇ�>\�<0y>���=�G��:��<�T�=�L�=<�/>Z�=��<Iʂ=7g�=�Q�=c�L=N�n��.��l�K��yc=~�л�«<��X'�<]ǻ�|��Ҍ�#�J������3=�x��J���g:;+*;���H��ϙ<=>�<��@������K%=uU=]Ǖ=E ���Ƕ�{�<k���d���9'=��޻�_=:�<\��E�<�^�=3����2�%#�=&�ͽ�_=g{n��!�%��=��t��/ļ��]����=7l�:H��?�=]>�菽�&����.=Q��c߼D�vu�Z�(=�"�����t0�ϏO=��<��;k����=��>�2�>���=��߻$$?�_>/<?�2>�WB=���� ^9>�T�;�����>6�=RF�>��>���>%���g>s��=b\>(�=M��=E\�=� H>D��;�D���<�X	==��<c��=�P >�q�=��/>wc�=�>���=�n$;{��=���<�\<�">U)��0=="}=mW�<��=.�+>��T=���<�ѐ<,��=�s:=l=k ~=�i�=.1	<6�X=���={��B�<CR�;
֓�+)�=5�e=U�뼧�����<��t<��i��"-=o#�����<��=�>�=aĚ=sF�=ou��}G���'=��]��ӥ=���<^�.;�鎼�����3U=�͕<K��h�B=��u<�n�;����,=

ٽ���ZX>�����M�Ț�=9-�nI&>�;ƥ>B��=$�E��#��u�=�xڼ���N5�=���k@= ��=K�=��U>(%O>�a����>�u�>7�&���b7�޽�<��w��\>��?b��>��>�Lｼ-?�`?}\�=!�>�HF�`>̞>Vm+�-��=$�?a�Խ��>�ͫ>Q��>��ܻn[>� �>M�?U½>�4����>�jD>��������>T[�<�,�钻<.�^�v����=�a�<�y����<g1s=���<�B4<�s`�3��<�=A��x!=0!��5`�7Ik=���<aH����/=c� =\w2=�$�����<i�ɽ�����==Y��<�w��4%��㈽B����1�=0|����j2�ot�=�!�wv= �ͽ�oe��=�;)E[>�ۋ��7Q��ZQ=Bd:���ƽ���������=��r<D>!=$=Zѫ=0Wa={B<^��c�=�m<�=�n�<�q�)�5�Zhq=�4�6ʻ��)�?�<g��=� �e����8�XU����n<�o�<��=�E���|�c��=�$c<�=Iq=�=�-`=H��;��=ۜ�=4!�c8�BJs=F��=PwL��W=��Y���=�9��*)��9�> �������1�=����O��=
�<�E>x[j< �C�h雼�U�>~ٽ�ꆾ�r>߹ʽ��Y��C�=�9�= j=�]B>�����KZ>i��=,w�<�$�<%X��� �=[�O=ׯ�;ɩE=C�$=^[;>�0�硥�n�*>�l<v^�=�5�=�v�=���=�=A���l�=�ȉ����=Y�=J������<���=.�=�i�=��[>�����>�'���J�<��=5>���>���=0(R<�H�>Tr�>,��=R��=��>��N>�H����9<0r_�����c >!��<�x���ֵ>�2{<ٚ>L�>��>(�<��=F�R><!�>���=�=~�=]��=K�PuԼ�d��W�=`O�<0k>��X=�1���>șN��� <x���~���>�:=�X�=rn�=O=������=�و��曼�$ =3�t=��B�Q<ˤ�=�5=lV�=&=��N>�=�)�<�T�=�� =y]d=���뮉���0�ݤ>K�<����AȻ��ҽ-��R)<0��O}ϻ]-[�d0?����w��;�)��^�����F3=(V�#'<]I����<�C�p��=c�mĩ�Z#�=��;��ì�ښ�=�=�=�vi=��<�\�=�D�=G��=?��=�s�;��G_>(nx��ǎ=��I>�ŝ<�FX���>^1�=~=>���=�<� =桀=�MT=䜾=pc�=��2;vv�<���<�=c�<���W���hؽ �=�'}>���=�
>������>�hL>g�S�m��=ѣ�����<�>V8���������>���:Jޛ>�k�=��>p�O�:P�=K��Km�>�]S=%�K���>�=5~Z�Ѩ���G���#=y9:=a�=�<l�����= ߏ=}�=3j�<�9M��� >8G�S�s=~a�=̕��X¼��N���нȓ��J�=���=;)=^��=��=�i=�@:u=��=�D=O��������)��i��l���k=��5��+�M���#3���0���Z��v���=��s��S�=�`P��{�<�o:=Vb�<�a����=\�����S=@Z�;g[�o������;h�l�0��A4���*�<Ʀ���k���O=m���E=�]R��V�=�3
>LǼ��=�=��Q��Q<=�:7��?='&=��!=�Ro=�*=�=��w�	=���=�
K����=A�<5"$=�抽���=l�0>K�`>-
3�a�j>���=���=K0>�<�։>����=g��=��}<.�=���=��"=���=+���>������3>��=e0�<�,=L�f<c��=->L���>�蔼q��;��E=7�=~�}=�d>F�]J#>�v;=ҙ�=�=i=�H���k�=�O8<�s=R��;]�R=W5=�Z��h��w�q=�a�����<�:Wt=8r<w2%<S),<�A,=� Q=(y���� ��p����<�YܻyW�� �o�mw�;�4���O��Fý`q�����$~=Kl/�����0�>.���/�#�m�>R�_�m����+=T|��u�=�>���>%	�<3m�O�=��b�ߨ�����=w�����=�"�=1^�=޺��^��=	����.�=�P�=3[�4�8��%B=X�#�֑�����s���JV� >�=�2��>�. }��
����i=��*=��=��h����ʥ^=��Ƚ��	@���C>�q�]�,=�2Z�Z�=5�׼�=�}�<z$S>T=)e2: ��m��<E��<���=��=֔����<��*��t>�砼����v���G#E=*�	=���=�`��Sj=W=�~��E�=M��=���=yՕ<�p�\�<��s>�	=����ɪ��!�=�v>C�=�q�;���;�O1>�y��՜�=A��<��b>L�Z=e�=��;)�	> �O�nJ@=&�=R�L=�Y>�+=U�7����=W�==:�=B�ػt��<#ð�y?�=��=���=�r>>���=�Ru>�� =6$w��C�=x�!���>g�	��v�� ��im��H�2$л�=  ����=����wj��N�=鯊=5����.<����ɼנ:��E�p�;���0�<ף�o}W=�Z������<��%�[kB��ӛ���7=D"='6���Ö<�Ľ���<PQf��L���=@Aغ8�d������+�<OpU����V���gd=G��:`��"�T<����ξ�^�������*~��]����<�x����}�}��8 h=/�=��=9��9���@��=cZ��U�_��>ѽ�~����=q��H�
=�ᇽIh�=�{�=n�'�b$��u �=�B*���н�}�==���=&��=='���p���=���<���=T�T=�ݽRU�^'
����=��ϼ��=Cz
=�̀<v�=��=�.��'>B)�<?1C=�3�e=�U<�I�=B�a<E�r<|�e=`�����_>g�O=��'���g=OW>�
^=��<7���i��=o�=ʑ�<�b> 6=Pc&���Ȼth�=8���;��>:�$��彦~ �篎��������z+X>�M[;�u��j5�=�冻Hx��(�����=Q����=A/�=P�F��>I<ň�<����t�=fS=v�߽k����R���M;=�:�<���=똉��-����=#������=R]E�g>8�c����=2�>�����a~��k(=mE�;ko�&>>�S��R@����=��>xա=�n�=�$��A�[>�ێ=K3۽7�K<F�=(<����j�6�/b�;ZB�<�"��I<�z<~OL�~X
=o�9��Y�8.�]�k�[=��q��)6�á�=�K6<���ә&=&��z��<e����x	����	��̹֟�Ҩ�TM��5�la��%=%�#>�����c���(;�Vнz��=���=U�޽5m&�s~��@ =��<����=�}���l6��NM=��#>��%�F����舻�ѓ='��;T9��^2Ž��:���@��E�=v=.	*=��#��FF>S��<*,>lK�>�?��>��=�̒>� ?L�<%W�>jm=q��=�i>��=r�Z����>�#ɼ�v4>���=�U�>�pf<d�>�T><�>��>o�@=X��>���>�Cz�C�=2;�k�=�[==o6�=۽X;<?]S>�ɱ=ZZ">p1O��S�=�+<���=�
>��!�B�:�༇�Ǽqý�"<�����}�j)=[�=�m>�=���>�I=�%ҽ��N�[��"�t��ʅ�ᑛ�[���Ԗ���&=\��ȿo=��z�p�Ҙ<=2G�$�=���=��ż�>�=]�_���Ľ+�'��N=�+�=��<��8<$Ӝ�L�<
Zp=m1���O]�=U[O�0�ݵ�B�Ǿ�n����>��i���"sv�!�t�iP5�B�J��q��@ =���<�k�=T[�<})����'�ؚ�=��o���h��D�=�ֽE�a<�h=��=T�>�뼇f�<x�=
>-���C��M��=�Be>��2��ѵ:�[>w�>��e<�<���=L�?�L�x�D>򃌼��>��>��ּ�Ͳ=薪>{ѱ��� >삣:!>n>V뇽 �,>���>IK�> ��>��&<+	�>
	�=���/��=ƞ�����;ڼ,J�<a��<��&��ƍ;Ê�<����!��ft�l��=H��)ܼ��Z����<�ؼ�;��l�� �=�ἂ3=	�Ž;1����;y�ռ���<�+�=`@�<b-�BiP�l��<�z��r�$��<�>�<�T�@�>b��K�B=W���3`�Y���>�<����=�Z<Z�(�`<t/�H�����	���*�`��=g�x=i_�=��s=��=�^R���<7���l*�=�B��.\���=6��``�<��O����H�/��=*�= �<b�<�>��=����;zж��zl�qJ�%N�<>&���>.�ɼ�PW=�5=�{8=τ�<�	\=4'��De�<(�G=Z#x=S��8�<�F6=l?V��n`��=��.�}���O�p=�8t���Z���SP���k�<�6=>���D��SR;W�/<�*���@���r�=�����
����*�>\g8���;[�x�c�=�� =os��5f,�B�X��퓽����ǵ�=���7��r=�{H�ֵ�Fwl="��0F>8����>�uf<m���2.����.�*��h���Y>����?�8�����⽽ny<]˭=�g���&>d7>�c���wm��*�;4�>QWm=4�=Z��>�]�>p������X$	>jZ�>��/� x=zo��nf�=���>�X�=�t��1�>qdn=i�w>|L���>`1�62�=�H�=���>8�>���=��>�^�>���=��d<Z{7:&�O�v�޽��9=߫';���U�=n�����@�A��@J��++=ㅹ�\4�=��=���T�<�H��n��iQ��7=>�G�����K����=^�=j�
>M��<t�&=�:�=�c�cWw�0U�����"����<=
����0��;�I�<�s���ҧ=�<�}�D-�����(�=*,��/�<�ˬ=�+=$;�;Gة:��0��D�;?),<J�>���=�I�<�}�j�=Ě8=��<��";��|;"<=4[�;h:�<�)7�Y�z=�`�=�_==��s(=	P�<q>/�����,>��>�����r<�
�=��z��c�<��>�
h��S=%�p=-�1<9��<�>�1=�Ǭ=Җ>xٲ�g�=4m�=��.>�ˬ�E,>���>���>W�-=,j��H4>#��>$�U�>}�=q�>�jP>�7�a-��]j�>��H=��V>��=�_�>-U=ۯ�=_�d>j��>6�v>�S�={��>�>w�\q����G����;т���=]���9��4��=k��".׼"z=�)=x#=��6.v>�s�=�e"�x�7����<�!޽�+/��0=){н����j$�=	Y�=g-=@��=N�n<I�V;b�=����m"����`=��<��E;�'���\>2=4��=��ƻOϩ=P�<�C �M�I�*�5=k���F<��<�-���ϼ�4�ü@7��x�8<��ؽU�>=V�e���G<�t-�x��}����D/<s�=�H<�����-���I�Z��=���<$Ok�(�b��W\��m��26o����;��=L5]�dҽ=Ze�=��5���L=��<t�������=2k�/Q����>�j�=���=�>l���_��=ޒ�<v�ꝙ�2G
EStatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp�
EStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
: *
dtype0*�
value�B� *�4�>9Oǻ�\Q=Eo<�0x�9>�ܼOX��H��=�5�=����}A>�
���Yj���l>16>����.�^���;Q?̼g���s㦼����jn=��>���=Oq�3>.�g>�3>���Yk=�4�=��S��lm=9�*>�qF>���=���X�U�`S�=fX>2�=˴>�&F�_�P�[��<��>Lֻ�0Ļ��8> �3�v����[#����=��>��>���=-<G��=OF=���=X��`>����|@���>>��=��<������[[>����=	;;xo�=S.�<{�t2�o�����<�-��9�>��M��j�;�y>�pL���>亏=���Vj>�ϰ�P�F<�2Z�H	>E|%<,��=2G
EStatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp�1
EStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
: *
dtype0*�0
value�0B�0 *�0�M�ʷ=��g=�嵼jf^�r�@��+׻��~�3�y���ļ�S����ռC蚽�(;=:��Wt;�e�ý�h���%<};׽�8�<\!���Ư=$�C�K�>ʽ�=Ф=eځ�X���=1`�<��J>�*n�\�=NC�<�s	�'=,=��&���=�S=d\>2�]=̠{��������<�E� �t��t1> P<jj=v�[=�G�=��/=O�$>v� ����=�s�1�9�|�=բ:=���=��=���;� 2=٢Y=��3=�v	��<Wp��h}���:>�R�=��5�g����=Qz��b@���>��;��I�e
�;�����
=�D����(�0W�/D>�[:���`>8=���=�y;<@���jݝ<����G<>h+���ѿ=���=���=e�<�	�7>�hH�iο=�B��2�J��k>�X=lp�=_�>���<���P���*��d�J	#�\�U=�>,=w��=��=�j>==�W�򺠻i�!��S:���<˘>���~9�=ݢ	=�j�=`����t�� �>�h�� ��//�=��=�o=KQ|<s�>mx�z��'��=�9c=���L�!>P?k�G�=Ɩ��L�]��k��y��=�4n������>i�(�;���J���;Q=�7�B�{<em >���=}���k��v���=ڭ��cƼ��g��hR<��~���<jA*<d�=؂�����=_�Ľ
e=׀K�$Q7=����C����;�fT=2�=��8>�g���_�ݫm=3��<����'+��	�<�F��'�=R/=����;gH>��+}�m��=%o=|�:*m������>�~���,�=B�';cm>�$���A>z�=s�$�H%b=��(<���;0�>/�8=��g���պ�]�=w2�C{�=mQ��ޠ�<�¼�*�={�սuT����>�<�zY=���>�B�����ν�׽{2�"����;�fȽ�>D�-</>�q>��V��$�p�=�#�Oڰ<��4>��<�������1G�;���C{�=��<c^S���=O}ּ�=E6w=[����\L�	����q5=r�Q<^�=c*G=��='�b��Il��x{=G�;.���f�����P=U�ѽOW=K4�>�]<Z` ���9��T��dO��w��zm&=���]8O=�e���T>�"=��C�4� :�6@�A�a���=�Na>��:��n=.?s�֣�=��ݽLq>"�V�u`�=�ܼtӦ=A]��pS*��S;;ND���c=y��p�T>Dvv<��+>�O�M
�=�l���_={炾�6=�=�4�a;?Mb>�W�~xK��X�<Lk�<X�q<�*�=��ƽ�N�<c�@>�5#���8�'��<(��=F���A����>;�L��
�<VX�;*	u=�V<%ü?��;�)����P�mZ�<
F��Ҷ="k�����=�!��<ă�6=މ�='e=<G�>P����=D��Yj>������<�h��3[��<jV�~������<\R>s�1=mEo=���v�=���:�{�:2f޽c�a=ۘC��ݵ=߶=��<��=�[ѽG/>7@x>+px�L�Z<y�мJ>Zռy�l>%�ѽ�`�Hh����=-�����=!	>S�k<ܲ�<ȫ�>}��=�}����S�)J>��;;�=L�� ?�<1
��0ې<#�|��a����m_缳=O�g>2�P�<�Y�<nn�=��;μ�=�.w���=�=_��;,�=#fh������<���<]��9�c潤�<�|a<��=�Ӻ<(G>i�u����:y�=\&c����o/R>���=�6ս��>oEM��IG��Oм����&�;)�6BM��N���mdB���?>R>�a湰�3��`����/>}=V=�=�fܼ�b{=�jh���l{�<T����P=DD�Du>>	�#=Q���[<��,=z��L =f>�j>6�.�F��ݾ�w���I����=,@�5�L>X�<�Y��������b���#w<ɯ=�5Q��j�=�6�;E��=��=mu#���e={��c�7�O�8t�>l�<o'�=K�=C����O��ec=O��C��gMm<����=��)�=��*��s&=�T�=��>t�b<�S���kn>hc��3�=`%t���b���<���<�< �f��<��o=�p�=���=�F$=�kӺ�l�G�<UyC=�m�}u>������<8���<�潐�6=�&�=i�Z���j=�>�t��q��<�5���>L\��.=�=0��<�!4=�_��P=GA��P�= q���;�!=8X��N�=[[�<}�>�g3>C��<���<�:�=J��Kɼ9���\�>�6ͼcm
>}��&:6=��=B�>U�=�DQ;�n>L-�;�->=ֱ�<�=�<����#X=�?�:�=�ߓ�t?=~�佃�$�����o��;�E=�ɑ=�O�g����= )��(���t��$�������==���<@i����'����9w��<��Z�џ!>W+ =N��<1��̾���>�"R=6}i���3�KW����� <>I�����>Ӻx���n�O<�;�X�=� �h�l�=�B=�^Y��0&��
�=ɷ�<�ܜ<c���S��r�1=� x<��a�&�(��	X=��=^h&=[�Ѧ|;6�ջ�y��E2�<9�O=����4\�1s��.��:᧿<�V���~V��V��}$>`H����3�������9�"��q1>���Q���.7=�T��Ex�=�s6��B;�����7�SR�:R]��(&��g+</��`��)7�IF�<hI==@}�m����D=oYM=�4���P�I�q��e>�Pp; �R�N��<Y~=/v]<=Z�<π�=�u��H ��`B=�D��'%��oc<�b(=�ٻ��=|ώ:���= x�=��>u+��"�=ѹл��'=5��<������=&l�Ӗ����=��V���=��S�=�C�(:<<.m�<�|�=PP}�I>*_���ev���<bK�]7=ם)<��r=�F=b:����]����=��d�P����<�.�=�<`딼lJ�����=h�)&<9��y誼�;=l>(� }�<M�S=���=��%��"�=9`=�[L���<P�K�d(�=w��=�N�<�[>�P��� =�zѻ�dN<�����@��<����=�.�=�&��7˼^+<ʴ%=�OP�$��L�=�?�/̱�.�^:�f=�E�kKQ=�y
���>��=��=m���Q
��^>>%�F=!�=�N���G>b_�M��=�eA=64�=@����<q%������b=�X��%6?=��2=
�����ܼV2�<)�
>�J���=W}˽�����1�<s(�=q�6>��Ͻ�*�=�6���!�Bj>���;��=�K��a<���ڽ�Ө��O"<[ν�б�A����^=mz��(,�=��+�=��=�u(��R%�|ɿ�4�.=+�2=�8��렢=��2;3g>���=dN�=�_>=�%�=��=i�=���=��=�y�Բ=�V��e��=���=a��M�=pP�=%���>0l�=]�=����Z��A�=�ue�9�/�=��5=�[������t�Hɒ�h;�<0�:��+�=���=��=��>�+���P=���<P�<ڀ��d$�:t>~V�<|U=��;Z>9c�M��=*�k�<R�[�)�h���k�<6�<J�c��N ���<�;�<�Z�Z~f={�=�UX=7pd=��;V��=��-�>�?a_<I>�=���<k�f��&��μ��n=�aR<�A4�9�9=8X=� �[��<��P���3��Ck��K�;8mý6�� ʨ<_AC��;��A�ƽ�g��b���@��=)o-���a>%[½B�)>)�=�[�=���p*��K>�m�<�N>v�W��%��������ȼq���������=L���S�=���&����fͽ��(�0�Ǽ���q'�=�=������o���:Ja(��ә<��=Y!d��Ȱ=Hb)>�R�v�E���=<ؔ�=�[<��S=l�<��"O�=��<�ϽP	��m��=�d7�3�<���%)O��p�=�ք=���=�-=�_�<��5�n<������=�\G<!��<=i0>�~ ����=�¼=���=h�;,lj=�I>`�"=��4>�z�����f���U>�d��hyZ���o�{a=>�S�=�y=�MY�Ϸ%��%H>���5�>�}�=�\�=��u<mM��'�=^���ޘ�=k�> d���Kͽ>վ�A=?ɣ<T�;?ԩ���~=U�=����S���Y�$>	#>�Qi>�i�>�BP��н�@��Z����r=x���d=Sh��C]d��w��ʭѽ&�=���=�$�=��>x�c<r}�=��=�]4<�h-��G`=I�C�>z��|�!>s�5<��=A�e<	�	�Ͽ��z�=�>��9=y񀾀8 ��|�Z+=��=*��n�=gc4�uE�<�VT=�y���x�=ȟ�I���$UB=l�����=�m�=�A�=F�<�A��J>���<��>� n��/�=$@��M½[�->K%d=��>���=2[�c���/ս��:�t�h�N��q��+����ν��t=��p�#pȽ��!>dFv=��>�3->.�i��t ���A=����}�����᣻Tf�=�u�<�	=H��EPi=��=�.��ě�<0����M���
���㾽�Vpź�B=�T�Ԃ��]ր=t�=>��aۦ<0�=R�F>�=�A��bv���ҽ}&�?��=�,��cm��x�#�'<��=Zah=��&����m=D>;;W�<<H���$ϼj]�1ڒ=4�;�K��;�B���<�����-���&=��b��i=�j=���>�=�u9�'�M<�o۽�ԽX���~C>!E=�a�<�0<�|?=?ټwL���K>�
>��>�X>�c�d�l�5_�� ����!��� !���K�ۼ-�������}=�\ֽ�g>�QP=���g(>N=���=(ݎ�0�5=4�>�`�$R>��z�p�=�@�=���=+��0޹�%}=+��<E�,=�һ�=��t=����v|v�T��<"%����9�~<�ZM�c[ѽ/$>Ơ;Y{L>�^1>&5�+��Nn��3��14�=8�n���+�jj��;9<�K�X��<�>P��l�*:`5����	!�=��b<��=j[�=�v�6}�>�5��m�>(:g���=Z���Yq�=��=���j>V�P��X>^�=zp=�鱽�1�<U<=�;<�Ka=	(�����=/8��Fϼ�
>�l�=[_�=�*>��@��/ý��/�ށ}�w��!�\� �����y�=�
+=6��<B�K�|�>��=Ƅ;>�x">��=�Q�=�q2� +`���=Tt��֖�=.��=BqѼEd!=���u{=�,�;�^A=��=<g K�_�L��n�s�;d�=�jW=�4ŽK��)�,</*=��=M���{%���=$ř=\V�=��>�Z=[�=��`��u��:�2d�o���0(}��ZF=]��=����v۟=|R=,F>���=�d�=̓ ������꼦t�=���=~��<�n���_=���Ņ�o"6>In=^0�>�-�<�$�<E�����E���O�r�a"�<�_�vP��=\~�=׉���/��(Y��[=��c>d�>Z�1��;�\��<G�<�0&��)�9m~���#=/*���=	�O��R\�ڒ�H
�<PX=�;�=�AM;��T��	޻j=�5�<��c=[kg=E��Π���J}*=��1�'��>��>ǔI��0��[�"�Փ~=�`<WC�:f�=aå����Q�:{��<��:���F�eh[>�l��1Q�=�t=؋=y�;0VW�K�c>���<BqV>�܌���<�l��PX�=;-Ľ5�ͼ
tq����<��b>x��=,Ŷ=Ø�[7!;=�
>}� ���->$���d>�8�Ө����̽	N��<o�J�&@%>�,=u��='�=�{�=�W�=����4P>2G
EStatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp�
EStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOpConst*"
_output_shapes
:*
dtype0*�
value�B�*�M��<\GR=�W�;ŵ�w��=`����|Ի���m������i�x��x�K|�:+����'=������3=S
<'p��c�=��<GA=T�ļ� (��"�<Dm˽���V?>�m=q�>�KԾi�=ѳ4���W�%�����@z�ѽ�<�<ɗ��Ž�<w~�䬼��<Omr���<�aJ��'��E�=�V�*ٯ��<�z��r��=�m����qD�=�˽��F�W�l>5=c�=����V�����=c�m�|#̽@W��-	�=;'�Xk}=����E��{k =���=f�V>��&=�սk����=��2=a�E=�tX�S�	=T�<�����!<���+�^=��ʽcǣ=��(>w)=�����n��ں<۩���1<�]�;�hZ�����R�>��=��B��X>��X>�	���󅾰�=���<�Ά�L��<D������ܻ�=��Y=�����J���p�����F�l�	=�=C�%���k��������=4�ǽB��=*�:=��3�:��<
0�<�ᢼ�W��7�����|�^ĺ�4þg��������>�8.=r�=��>M[�=c6�G�'<,���� �=�]�<a��<�)�<��I��������P�������=`�^��0����=�se�쁽3�I�\(��~1�H���:J=�)d>~� ��+>�y��g�;�)<)k�����.�o�;�<(�<�	�%a�<����S`��[�=Qd�=� >U=̊����R�Ȃ=O�-<@Ly���?<P�н舄<tuP��_ѽmZ$��ս�?���c�=-��;�ذ=����<z卽��W�=:W<�����/��4L=�;���7=OM�����=YI�����>���<2f�=+'��O����i�e�=`�<�{���?�&!�=A�?=א0�7��<.޷�)'�=.%���̼�ֽ�q�;nH�A
(���=��*=�ŝ�0�:=����@I���S=WP��^�>���!ڪ��	
�����h>J�s��ɽ�Q��;�l=�:ؼ�a�Tv�p~<�W�l��ԥ<�P��ݜ�6��<Xm��=�?���n��b\�XY�= so�9B=�<�6�=}�/;�֘�'���#��8=f>���u<[�x�q���=�-ٽ�3���`=�t=���Tf�=K^B��@���#B�п���=Z/��	�R<o�<܊<�d�<�BI=�zλUcN��
�<�VĽ�r�=a�5��ӝ���ý�v�X��=���<�d�=X9��Q���_R=W�=A'=J�>d�x������ه=�����{�TQ�>�.>�J�m[�$,��s"��X��&�2��)=�h��ذ<���=r
�<D!���<H�N<`�9�� �^���'4�=0����
��=ڞ�òS<��>����YK�܍���=��<����_I�>Q>D؉���̾Z?�;"K}=��ݽ>��:�&�<��;�^�=���=#��=���=���@Y���->��2�����=J�G��� �=�2ͼ�!��$e�B�?��"�<|.���;��Ze��O$=9�ѽ~���>+ʼ�Gr�&�b=��=1����Y`���;'�p�oB��o|��鷵=L�q���=�zW=ӏ��L)��E�>���<���<ĪR=��J<�����(�eU�<�n�<�v����`�Żj�N�	��;�ۗ<Aؾ<�F��CO��N�l�.�=���<au���Ӏ�nT�=h�=pI;�q�<
qf����I鼔ٻeFR��f=�̫��5���O����½��.��]=a�y�:��<O �M=殶=���/}�%��������h�=mbz<-���p�=�%��`\_��@�<R|'��|�:��x�8;��G��8�W�����<wc�=Fب��0ý^�=��̽�O�<�;i�3��[+=#���	�7ۼO�</�W=��$����=<�s=���R>���d���e������v�;�Rv��{�$ُ<}��9p<����N+۽M�ڽB���d�<=��<y'�=�ah�2���Lj�'c�QI����M�n��<������<7�~=b�=Lo97�O=��^�5r��<*y�=��$<x��=��>8� <��=P��;z�Ͻ)o�=7!�=CA�:h����=Y�=�d�ƻ�=xS��4mE:� �<5.���Kz=[Gú>���⼾'
���ǽ�Ȼ��l=��y�΍=6Rz��)�=Y�Լ���=�G �X��<���<�A�=�=GD�-�=Oÿ=���<b�;D"�ZY����������1=�=ip>�'�e�=|G=�D�2��p��%Հ�J=����Ҽ�]B�M��b�>:zE�5��<�et=ou=�s>D<>>Ml/=fP=(�=�e~�X�=���eA��%��=�x(��-=�7���'�
*�=�_�=�K��MF�V��� ��KV��v�<{n'=�Co��*>C������Q�>pj�����=)�n=���=�+�=�>9�=o��<mଽ֠�<2�=)�½����6�>�灾�W.>�B�=F�v>_1#�Q���H<��q�� ��潠
��򅽠�P������k>;�y=�9�=oB�<u� >MJ>���<Q	ﻝ���(�=Z�ڽJ�o���i�VhB<4#'��� >�M>��
����<+=��>ר[����=�G!=m��=	�I���p=��=
��=X^�B�o>�7�=!u���=�F�M��<��^���h=����d0=�q˼�WW��^�sa���R�r�=��n=Y!�|�Ƚ��7<�jG�^%���,ڼ휆=���=�=�"�=lg�=;�Y���U=m`)���>�C=�=$�̼Y?8�ed�=f[���x=n� ��J��R��<Nbu�p^�!����k�����;Ts_� ]l=�	=�rR��}��?	˽�.�G��]����(�	սUl>��G=�ڿ;�3�<�)����k;��tO�<͇p���=C��⋼�@��9~Z>$�̱½K�\<�͓��Ge������c�<C��<�6�B�=w��>ȅ�=V�=<���$�=2G
EStatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp�
 StatefulPartitionedCall/IdentityIdentity6StatefulPartitionedCall/mnist/output/Softmax:softmax:0:^StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp<^StatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOp;^StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2"
 StatefulPartitionedCall/Identity�
'Func/StatefulPartitionedCall/output/_11Identity)StatefulPartitionedCall/Identity:output:0*
T0*'
_output_shapes
:���������
2)
'Func/StatefulPartitionedCall/output/_11�
4Func/StatefulPartitionedCall/output_control_node/_12NoOp:^StatefulPartitionedCall/mnist/fc_1/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_2/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_5/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp:^StatefulPartitionedCall/mnist/fc_6/BiasAdd/ReadVariableOpF^StatefulPartitionedCall/mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp<^StatefulPartitionedCall/mnist/output/BiasAdd/ReadVariableOp;^StatefulPartitionedCall/mnist/output/MatMul/ReadVariableOp*
_output_shapes
 26
4Func/StatefulPartitionedCall/output_control_node/_12�
IdentityIdentity0Func/StatefulPartitionedCall/output/_11:output:05^Func/StatefulPartitionedCall/output_control_node/_12*
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
�
�
__inference_loss_fn_3_2759064:
6fc_6_kernel_regularizer_square_readvariableop_resource
identity��-fc_6/kernel/Regularizer/Square/ReadVariableOp�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_6_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
�
�
B__inference_mnist_layer_call_and_return_conditional_losses_2759289

inputs4
0fc_1_conv1d_expanddims_1_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource4
0fc_2_conv1d_expanddims_1_readvariableop_resource(
$fc_2_biasadd_readvariableop_resource4
0fc_5_conv1d_expanddims_1_readvariableop_resource(
$fc_5_biasadd_readvariableop_resource4
0fc_6_conv1d_expanddims_1_readvariableop_resource(
$fc_6_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��fc_1/BiasAdd/ReadVariableOp�'fc_1/conv1d/ExpandDims_1/ReadVariableOp�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_2/BiasAdd/ReadVariableOp�'fc_2/conv1d/ExpandDims_1/ReadVariableOp�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/BiasAdd/ReadVariableOp�'fc_5/conv1d/ExpandDims_1/ReadVariableOp�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/BiasAdd/ReadVariableOp�'fc_6/conv1d/ExpandDims_1/ReadVariableOp�-fc_6/kernel/Regularizer/Square/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpo
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
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
tf.expand_dims/ExpandDims/dim�
tf.expand_dims/ExpandDims
ExpandDimsflatten/Reshape:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims/ExpandDims�
fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_1/conv1d/ExpandDims/dim�
fc_1/conv1d/ExpandDims
ExpandDims"tf.expand_dims/ExpandDims:output:0#fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_1/conv1d/ExpandDims�
'fc_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
fc_1/conv1d/ExpandDims_1�
fc_1/conv1dConv2Dfc_1/conv1d/ExpandDims:output:0!fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_1/conv1d�
fc_1/conv1d/SqueezeSqueezefc_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_1/conv1d/Squeeze�
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_1/BiasAdd/ReadVariableOp�
fc_1/BiasAddBiasAddfc_1/conv1d/Squeeze:output:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_1/BiasAddl
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_2/conv1d/ExpandDims�
'fc_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2
fc_2/conv1d/ExpandDims_1�
fc_2/conv1dConv2Dfc_2/conv1d/ExpandDims:output:0!fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_2/conv1d�
fc_2/conv1d/SqueezeSqueezefc_2/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_2/conv1d/Squeeze�
fc_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_2/BiasAdd/ReadVariableOp�
fc_2/BiasAddBiasAddfc_2/conv1d/Squeeze:output:0#fc_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_2/BiasAddl
	fc_2/ReluRelufc_2/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_3/dropout/Mulo
fc_3/dropout/ShapeShapefc_2/Relu:activations:0*
T0*
_output_shapes
:2
fc_3/dropout/Shape�
)fc_3/dropout/random_uniform/RandomUniformRandomUniformfc_3/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
fc_3/dropout/GreaterEqual�
fc_3/dropout/CastCastfc_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
fc_3/dropout/Cast�
fc_3/dropout/Mul_1Mulfc_3/dropout/Mul:z:0fc_3/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_4/ExpandDims�
fc_4/MaxPoolMaxPoolfc_4/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingVALID*
strides
2
fc_4/MaxPool�
fc_4/SqueezeSqueezefc_4/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
fc_5/conv1d/ExpandDims�
'fc_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
fc_5/conv1d/ExpandDims_1�
fc_5/conv1dConv2Dfc_5/conv1d/ExpandDims:output:0!fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_5/conv1d�
fc_5/conv1d/SqueezeSqueezefc_5/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_5/conv1d/Squeeze�
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_5/BiasAdd/ReadVariableOp�
fc_5/BiasAddBiasAddfc_5/conv1d/Squeeze:output:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_5/BiasAddl
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
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
:����������2
fc_6/conv1d/ExpandDims�
'fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
fc_6/conv1d/ExpandDims_1�
fc_6/conv1dConv2Dfc_6/conv1d/ExpandDims:output:0!fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_6/conv1d�
fc_6/conv1d/SqueezeSqueezefc_6/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_6/conv1d/Squeeze�
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_6/BiasAdd/ReadVariableOp�
fc_6/BiasAddBiasAddfc_6/conv1d/Squeeze:output:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_6/BiasAddl
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
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
:����������2
fc_7/dropout/Mulo
fc_7/dropout/ShapeShapefc_6/Relu:activations:0*
T0*
_output_shapes
:2
fc_7/dropout/Shape�
)fc_7/dropout/random_uniform/RandomUniformRandomUniformfc_7/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
fc_7/dropout/GreaterEqual�
fc_7/dropout/CastCastfc_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
fc_7/dropout/Cast�
fc_7/dropout/Mul_1Mulfc_7/dropout/Mul:z:0fc_7/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
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
:����������2
fc_8/ExpandDims�
fc_8/MaxPoolMaxPoolfc_8/ExpandDims:output:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
fc_8/MaxPool�
fc_8/SqueezeSqueezefc_8/MaxPool:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims
2
fc_8/Squeezeg
	fc9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
	fc9/Const�
fc9/ReshapeReshapefc_8/Squeeze:output:0fc9/Const:output:0*
T0*(
_output_shapes
:����������2
fc9/Reshape�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMulfc9/Reshape:output:0$output/MatMul/ReadVariableOp:value:0*
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
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
fc_6/kernel/Regularizer/mul�
IdentityIdentityoutput/Softmax:softmax:0^fc_1/BiasAdd/ReadVariableOp(^fc_1/conv1d/ExpandDims_1/ReadVariableOp.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_2/BiasAdd/ReadVariableOp(^fc_2/conv1d/ExpandDims_1/ReadVariableOp.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/BiasAdd/ReadVariableOp(^fc_5/conv1d/ExpandDims_1/ReadVariableOp.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/BiasAdd/ReadVariableOp(^fc_6/conv1d/ExpandDims_1/ReadVariableOp.^fc_6/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2R
'fc_1/conv1d/ExpandDims_1/ReadVariableOp'fc_1/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2:
fc_2/BiasAdd/ReadVariableOpfc_2/BiasAdd/ReadVariableOp2R
'fc_2/conv1d/ExpandDims_1/ReadVariableOp'fc_2/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2:
fc_5/BiasAdd/ReadVariableOpfc_5/BiasAdd/ReadVariableOp2R
'fc_5/conv1d/ExpandDims_1/ReadVariableOp'fc_5/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2:
fc_6/BiasAdd/ReadVariableOpfc_6/BiasAdd/ReadVariableOp2R
'fc_6/conv1d/ExpandDims_1/ReadVariableOp'fc_6/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�h
�
"__inference__wrapped_model_2758995	
input:
6mnist_fc_1_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_1_biasadd_readvariableop_resource:
6mnist_fc_2_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_2_biasadd_readvariableop_resource:
6mnist_fc_5_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_5_biasadd_readvariableop_resource:
6mnist_fc_6_conv1d_expanddims_1_readvariableop_resource.
*mnist_fc_6_biasadd_readvariableop_resource/
+mnist_output_matmul_readvariableop_resource0
,mnist_output_biasadd_readvariableop_resource
identity��!mnist/fc_1/BiasAdd/ReadVariableOp�-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_2/BiasAdd/ReadVariableOp�-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_5/BiasAdd/ReadVariableOp�-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp�!mnist/fc_6/BiasAdd/ReadVariableOp�-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp�#mnist/output/BiasAdd/ReadVariableOp�"mnist/output/MatMul/ReadVariableOp{
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
#mnist/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#mnist/tf.expand_dims/ExpandDims/dim�
mnist/tf.expand_dims/ExpandDims
ExpandDimsmnist/flatten/Reshape:output:0,mnist/tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2!
mnist/tf.expand_dims/ExpandDims�
 mnist/fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 mnist/fc_1/conv1d/ExpandDims/dim�
mnist/fc_1/conv1d/ExpandDims
ExpandDims(mnist/tf.expand_dims/ExpandDims:output:0)mnist/fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
mnist/fc_1/conv1d/ExpandDims�
-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2 
mnist/fc_1/conv1d/ExpandDims_1�
mnist/fc_1/conv1dConv2D%mnist/fc_1/conv1d/ExpandDims:output:0'mnist/fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
mnist/fc_1/conv1d�
mnist/fc_1/conv1d/SqueezeSqueezemnist/fc_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
mnist/fc_1/conv1d/Squeeze�
!mnist/fc_1/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!mnist/fc_1/BiasAdd/ReadVariableOp�
mnist/fc_1/BiasAddBiasAdd"mnist/fc_1/conv1d/Squeeze:output:0)mnist/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_1/BiasAdd~
mnist/fc_1/ReluRelumnist/fc_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
mnist/fc_2/conv1d/ExpandDims�
-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2 
mnist/fc_2/conv1d/ExpandDims_1�
mnist/fc_2/conv1dConv2D%mnist/fc_2/conv1d/ExpandDims:output:0'mnist/fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
mnist/fc_2/conv1d�
mnist/fc_2/conv1d/SqueezeSqueezemnist/fc_2/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
mnist/fc_2/conv1d/Squeeze�
!mnist/fc_2/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!mnist/fc_2/BiasAdd/ReadVariableOp�
mnist/fc_2/BiasAddBiasAdd"mnist/fc_2/conv1d/Squeeze:output:0)mnist/fc_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_2/BiasAdd~
mnist/fc_2/ReluRelumnist/fc_2/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
mnist/fc_2/Relu�
mnist/fc_3/IdentityIdentitymnist/fc_2/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
mnist/fc_4/ExpandDims�
mnist/fc_4/MaxPoolMaxPoolmnist/fc_4/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingVALID*
strides
2
mnist/fc_4/MaxPool�
mnist/fc_4/SqueezeSqueezemnist/fc_4/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
mnist/fc_5/conv1d/ExpandDims�
-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2 
mnist/fc_5/conv1d/ExpandDims_1�
mnist/fc_5/conv1dConv2D%mnist/fc_5/conv1d/ExpandDims:output:0'mnist/fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
mnist/fc_5/conv1d�
mnist/fc_5/conv1d/SqueezeSqueezemnist/fc_5/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
mnist/fc_5/conv1d/Squeeze�
!mnist/fc_5/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!mnist/fc_5/BiasAdd/ReadVariableOp�
mnist/fc_5/BiasAddBiasAdd"mnist/fc_5/conv1d/Squeeze:output:0)mnist/fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
mnist/fc_5/BiasAdd~
mnist/fc_5/ReluRelumnist/fc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
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
:����������2
mnist/fc_6/conv1d/ExpandDims�
-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mnist_fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2 
mnist/fc_6/conv1d/ExpandDims_1�
mnist/fc_6/conv1dConv2D%mnist/fc_6/conv1d/ExpandDims:output:0'mnist/fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
mnist/fc_6/conv1d�
mnist/fc_6/conv1d/SqueezeSqueezemnist/fc_6/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
mnist/fc_6/conv1d/Squeeze�
!mnist/fc_6/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!mnist/fc_6/BiasAdd/ReadVariableOp�
mnist/fc_6/BiasAddBiasAdd"mnist/fc_6/conv1d/Squeeze:output:0)mnist/fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
mnist/fc_6/BiasAdd~
mnist/fc_6/ReluRelumnist/fc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
mnist/fc_6/Relu�
mnist/fc_7/IdentityIdentitymnist/fc_6/Relu:activations:0*
T0*,
_output_shapes
:����������2
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
:����������2
mnist/fc_8/ExpandDims�
mnist/fc_8/MaxPoolMaxPoolmnist/fc_8/ExpandDims:output:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
mnist/fc_8/MaxPool�
mnist/fc_8/SqueezeSqueezemnist/fc_8/MaxPool:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims
2
mnist/fc_8/Squeezes
mnist/fc9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
mnist/fc9/Const�
mnist/fc9/ReshapeReshapemnist/fc_8/Squeeze:output:0mnist/fc9/Const:output:0*
T0*(
_output_shapes
:����������2
mnist/fc9/Reshape�
"mnist/output/MatMul/ReadVariableOpReadVariableOp+mnist_output_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02$
"mnist/output/MatMul/ReadVariableOp�
mnist/output/MatMulMatMulmnist/fc9/Reshape:output:0*mnist/output/MatMul/ReadVariableOp:value:0*
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
mnist/output/Softmax�
IdentityIdentitymnist/output/Softmax:softmax:0"^mnist/fc_1/BiasAdd/ReadVariableOp.^mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_2/BiasAdd/ReadVariableOp.^mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_5/BiasAdd/ReadVariableOp.^mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp"^mnist/fc_6/BiasAdd/ReadVariableOp.^mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp$^mnist/output/BiasAdd/ReadVariableOp#^mnist/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2F
!mnist/fc_1/BiasAdd/ReadVariableOp!mnist/fc_1/BiasAdd/ReadVariableOp2^
-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_1/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_2/BiasAdd/ReadVariableOp!mnist/fc_2/BiasAdd/ReadVariableOp2^
-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_2/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_5/BiasAdd/ReadVariableOp!mnist/fc_5/BiasAdd/ReadVariableOp2^
-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_5/conv1d/ExpandDims_1/ReadVariableOp2F
!mnist/fc_6/BiasAdd/ReadVariableOp!mnist/fc_6/BiasAdd/ReadVariableOp2^
-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp-mnist/fc_6/conv1d/ExpandDims_1/ReadVariableOp2J
#mnist/output/BiasAdd/ReadVariableOp#mnist/output/BiasAdd/ReadVariableOp2H
"mnist/output/MatMul/ReadVariableOp"mnist/output/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
B
&__inference_fc_8_layer_call_fn_2759077

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
A__inference_fc_8_layer_call_and_return_conditional_losses_27590722
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
�
{
&__inference_fc_6_layer_call_fn_2759474

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
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27594672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_fc_2_layer_call_and_return_conditional_losses_2758855

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
:���������� 2
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
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�U
�
B__inference_mnist_layer_call_and_return_conditional_losses_2759525

inputs
fc_1_562038
fc_1_562040
fc_2_562043
fc_2_562045
fc_5_562050
fc_5_562052
fc_6_562055
fc_6_562057
output_562063
output_562065
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_3/StatefulPartitionedCall�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_7/StatefulPartitionedCall�output/StatefulPartitionedCall�
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
D__inference_flatten_layer_call_and_return_conditional_losses_27587682
flatten/PartitionedCall�
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
tf.expand_dims/ExpandDims/dim�
tf.expand_dims/ExpandDims
ExpandDims flatten/PartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0fc_1_562038fc_1_562040*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27588842
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_562043fc_2_562045*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27588552
fc_2/StatefulPartitionedCall�
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27587302
fc_3/StatefulPartitionedCall�
fc_4/PartitionedCallPartitionedCall%fc_3/StatefulPartitionedCall:output:0*
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27589152
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_562050fc_5_562052*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27590322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_562055fc_6_562057*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27594672
fc_6/StatefulPartitionedCall�
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0^fc_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27591112
fc_7/StatefulPartitionedCall�
fc_8/PartitionedCallPartitionedCall%fc_7/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27590722
fc_8/PartitionedCall�
fc9/PartitionedCallPartitionedCallfc_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc9_layer_call_and_return_conditional_losses_27591492
fc9/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc9/PartitionedCall:output:0output_562063output_562065*
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
C__inference_output_layer_call_and_return_conditional_losses_27594382 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_562038*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_562043*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_562050*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_562055*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
fc_6/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_3/StatefulPartitionedCall^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_7/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_mnist_layer_call_fn_2759555

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
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27595252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_fc_5_layer_call_and_return_conditional_losses_2759032

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
:���������� 2
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
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
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
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
{
&__inference_fc_1_layer_call_fn_2758891

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
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27588842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_mnist_layer_call_fn_2759540	
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
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27595252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
_
&__inference_fc_7_layer_call_fn_2759116

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27591112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
#__inference__traced_restore_2760451
file_prefix 
assignvariableop_fc_1_kernel 
assignvariableop_1_fc_1_bias"
assignvariableop_2_fc_2_kernel 
assignvariableop_3_fc_2_bias"
assignvariableop_4_fc_5_kernel 
assignvariableop_5_fc_5_bias"
assignvariableop_6_fc_6_kernel 
assignvariableop_7_fc_6_bias$
 assignvariableop_8_output_kernel"
assignvariableop_9_output_bias
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
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
AssignVariableOp_8AssignVariableOp assignvariableop_8_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10�
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
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
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2758897

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
__inference_loss_fn_1_2759758:
6fc_2_kernel_regularizer_square_readvariableop_resource
identity��-fc_2/kernel/Regularizer/Square/ReadVariableOp�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_2_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
�
E
)__inference_flatten_layer_call_fn_2758773

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
D__inference_flatten_layer_call_and_return_conditional_losses_27587682
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
�
�
A__inference_fc_2_layer_call_and_return_conditional_losses_2759138

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
:���������� 2
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
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�R
�
B__inference_mnist_layer_call_and_return_conditional_losses_2759708

inputs
fc_1_562124
fc_1_562126
fc_2_562129
fc_2_562131
fc_5_562136
fc_5_562138
fc_6_562141
fc_6_562143
output_562149
output_562151
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�output/StatefulPartitionedCall�
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
D__inference_flatten_layer_call_and_return_conditional_losses_27587682
flatten/PartitionedCall�
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
tf.expand_dims/ExpandDims/dim�
tf.expand_dims/ExpandDims
ExpandDims flatten/PartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0fc_1_562124fc_1_562126*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27588842
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_562129fc_2_562131*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27588552
fc_2/StatefulPartitionedCall�
fc_3/PartitionedCallPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27589022
fc_3/PartitionedCall�
fc_4/PartitionedCallPartitionedCallfc_3/PartitionedCall:output:0*
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27589152
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_562136fc_5_562138*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27590322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_562141fc_6_562143*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27594672
fc_6/StatefulPartitionedCall�
fc_7/PartitionedCallPartitionedCall%fc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27594222
fc_7/PartitionedCall�
fc_8/PartitionedCallPartitionedCallfc_7/PartitionedCall:output:0*
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
GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27590722
fc_8/PartitionedCall�
fc9/PartitionedCallPartitionedCallfc_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc9_layer_call_and_return_conditional_losses_27591492
fc9/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc9/PartitionedCall:output:0output_562149output_562151*
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
C__inference_output_layer_call_and_return_conditional_losses_27594382 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_562124*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_562129*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_562136*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_562141*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
fc_6/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_2759748:
6fc_5_kernel_regularizer_square_readvariableop_resource
identity��-fc_5/kernel/Regularizer/Square/ReadVariableOp�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_5_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
�U
�
B__inference_mnist_layer_call_and_return_conditional_losses_2759606	
input
fc_1_561721
fc_1_561723
fc_2_561759
fc_2_561761
fc_5_561828
fc_5_561830
fc_6_561866
fc_6_561868
output_561938
output_561940
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_3/StatefulPartitionedCall�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�fc_7/StatefulPartitionedCall�output/StatefulPartitionedCall�
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
D__inference_flatten_layer_call_and_return_conditional_losses_27587682
flatten/PartitionedCall�
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
tf.expand_dims/ExpandDims/dim�
tf.expand_dims/ExpandDims
ExpandDims flatten/PartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0fc_1_561721fc_1_561723*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27588842
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_561759fc_2_561761*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27588552
fc_2/StatefulPartitionedCall�
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27587302
fc_3/StatefulPartitionedCall�
fc_4/PartitionedCallPartitionedCall%fc_3/StatefulPartitionedCall:output:0*
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27589152
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_561828fc_5_561830*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27590322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_561866fc_6_561868*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27594672
fc_6/StatefulPartitionedCall�
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0^fc_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27591112
fc_7/StatefulPartitionedCall�
fc_8/PartitionedCallPartitionedCall%fc_7/StatefulPartitionedCall:output:0*
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
GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27590722
fc_8/PartitionedCall�
fc9/PartitionedCallPartitionedCallfc_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc9_layer_call_and_return_conditional_losses_27591492
fc9/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc9/PartitionedCall:output:0output_561938output_561940*
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
C__inference_output_layer_call_and_return_conditional_losses_27594382 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_561721*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_561759*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_561828*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_561866*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
fc_6/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_3/StatefulPartitionedCall^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^fc_7/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
�
A__inference_fc_6_layer_call_and_return_conditional_losses_2759467

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
:����������2
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
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
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
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_fc_1_layer_call_and_return_conditional_losses_2759099

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
: *
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
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:���������� 2

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
A__inference_fc_5_layer_call_and_return_conditional_losses_2759176

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
:���������� 2
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
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
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
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
'__inference_mnist_layer_call_fn_2759738	
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
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27597082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
`
A__inference_fc_3_layer_call_and_return_conditional_losses_2759770

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
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�	
�
C__inference_output_layer_call_and_return_conditional_losses_2759438

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
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
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
{
&__inference_fc_2_layer_call_fn_2758862

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
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27588552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
_
A__inference_fc_3_layer_call_and_return_conditional_losses_2758902

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�R
�
B__inference_mnist_layer_call_and_return_conditional_losses_2759657	
input
fc_1_561974
fc_1_561976
fc_2_561979
fc_2_561981
fc_5_561986
fc_5_561988
fc_6_561991
fc_6_561993
output_561999
output_562001
identity��fc_1/StatefulPartitionedCall�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_2/StatefulPartitionedCall�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/StatefulPartitionedCall�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/StatefulPartitionedCall�-fc_6/kernel/Regularizer/Square/ReadVariableOp�output/StatefulPartitionedCall�
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
D__inference_flatten_layer_call_and_return_conditional_losses_27587682
flatten/PartitionedCall�
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
tf.expand_dims/ExpandDims/dim�
tf.expand_dims/ExpandDims
ExpandDims flatten/PartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims/ExpandDims�
fc_1/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0fc_1_561974fc_1_561976*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_27588842
fc_1/StatefulPartitionedCall�
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_561979fc_2_561981*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_27588552
fc_2/StatefulPartitionedCall�
fc_3/PartitionedCallPartitionedCall%fc_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27589022
fc_3/PartitionedCall�
fc_4/PartitionedCallPartitionedCallfc_3/PartitionedCall:output:0*
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
A__inference_fc_4_layer_call_and_return_conditional_losses_27589152
fc_4/PartitionedCall�
fc_5/StatefulPartitionedCallStatefulPartitionedCallfc_4/PartitionedCall:output:0fc_5_561986fc_5_561988*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27590322
fc_5/StatefulPartitionedCall�
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_561991fc_6_561993*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_27594672
fc_6/StatefulPartitionedCall�
fc_7/PartitionedCallPartitionedCall%fc_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27594222
fc_7/PartitionedCall�
fc_8/PartitionedCallPartitionedCallfc_7/PartitionedCall:output:0*
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
GPU2*0J 8� *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_27590722
fc_8/PartitionedCall�
fc9/PartitionedCallPartitionedCallfc_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc9_layer_call_and_return_conditional_losses_27591492
fc9/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCallfc9/PartitionedCall:output:0output_561999output_562001*
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
C__inference_output_layer_call_and_return_conditional_losses_27594382 
output/StatefulPartitionedCall�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_1_561974*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_2_561979*"
_output_shapes
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
-fc_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_5_561986*"
_output_shapes
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfc_6_561991*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
fc_6/kernel/Regularizer/mul�
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_2/StatefulPartitionedCall.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/StatefulPartitionedCall.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/StatefulPartitionedCall.^fc_6/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:���������

_user_specified_nameinput
�
`
A__inference_fc_7_layer_call_and_return_conditional_losses_2759301

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
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_2759049:
6fc_1_kernel_regularizer_square_readvariableop_resource
identity��-fc_1/kernel/Regularizer/Square/ReadVariableOp�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6fc_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
�
`
A__inference_fc_7_layer_call_and_return_conditional_losses_2759111

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
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
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
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
\
@__inference_fc9_layer_call_and_return_conditional_losses_2759406

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

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
A__inference_fc_1_layer_call_and_return_conditional_losses_2758884

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
: *
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
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
-fc_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:���������� 2

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
}
(__inference_output_layer_call_fn_2759445

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
C__inference_output_layer_call_and_return_conditional_losses_27594382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_2758768

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
&__inference_fc_7_layer_call_fn_2759427

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_27594222
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
A__inference_fc_3_layer_call_and_return_conditional_losses_2758730

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
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
A
%__inference_fc9_layer_call_fn_2759154

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc9_layer_call_and_return_conditional_losses_27591492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
&__inference_fc_3_layer_call_fn_2758735

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27587302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
{
&__inference_fc_5_layer_call_fn_2759039

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
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_27590322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :���������� ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
_
A__inference_fc_7_layer_call_and_return_conditional_losses_2759054

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_fc_4_layer_call_fn_2758920

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
A__inference_fc_4_layer_call_and_return_conditional_losses_27589152
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
�
B
&__inference_fc_3_layer_call_fn_2758907

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
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_27589022
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�!
�
 __inference__traced_save_2760411
file_prefix*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop*
&savev2_fc_2_kernel_read_readvariableop(
$savev2_fc_2_bias_read_readvariableop*
&savev2_fc_5_kernel_read_readvariableop(
$savev2_fc_5_bias_read_readvariableop*
&savev2_fc_6_kernel_read_readvariableop(
$savev2_fc_6_bias_read_readvariableop,
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop&savev2_fc_5_kernel_read_readvariableop$savev2_fc_5_bias_read_readvariableop&savev2_fc_6_kernel_read_readvariableop$savev2_fc_6_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*x
_input_shapesg
e: : : :  : : ::::	�
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
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%	!

_output_shapes
:	�
: 


_output_shapes
:
:

_output_shapes
: 
�
_
A__inference_fc_7_layer_call_and_return_conditional_losses_2759422

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
\
@__inference_fc9_layer_call_and_return_conditional_losses_2759149

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_fc_4_layer_call_and_return_conditional_losses_2758915

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
ʃ
�
B__inference_mnist_layer_call_and_return_conditional_losses_2759400

inputs4
0fc_1_conv1d_expanddims_1_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource4
0fc_2_conv1d_expanddims_1_readvariableop_resource(
$fc_2_biasadd_readvariableop_resource4
0fc_5_conv1d_expanddims_1_readvariableop_resource(
$fc_5_biasadd_readvariableop_resource4
0fc_6_conv1d_expanddims_1_readvariableop_resource(
$fc_6_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��fc_1/BiasAdd/ReadVariableOp�'fc_1/conv1d/ExpandDims_1/ReadVariableOp�-fc_1/kernel/Regularizer/Square/ReadVariableOp�fc_2/BiasAdd/ReadVariableOp�'fc_2/conv1d/ExpandDims_1/ReadVariableOp�-fc_2/kernel/Regularizer/Square/ReadVariableOp�fc_5/BiasAdd/ReadVariableOp�'fc_5/conv1d/ExpandDims_1/ReadVariableOp�-fc_5/kernel/Regularizer/Square/ReadVariableOp�fc_6/BiasAdd/ReadVariableOp�'fc_6/conv1d/ExpandDims_1/ReadVariableOp�-fc_6/kernel/Regularizer/Square/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpo
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
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
tf.expand_dims/ExpandDims/dim�
tf.expand_dims/ExpandDims
ExpandDimsflatten/Reshape:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*,
_output_shapes
:����������2
tf.expand_dims/ExpandDims�
fc_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
fc_1/conv1d/ExpandDims/dim�
fc_1/conv1d/ExpandDims
ExpandDims"tf.expand_dims/ExpandDims:output:0#fc_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
fc_1/conv1d/ExpandDims�
'fc_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
fc_1/conv1d/ExpandDims_1�
fc_1/conv1dConv2Dfc_1/conv1d/ExpandDims:output:0!fc_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_1/conv1d�
fc_1/conv1d/SqueezeSqueezefc_1/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_1/conv1d/Squeeze�
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_1/BiasAdd/ReadVariableOp�
fc_1/BiasAddBiasAddfc_1/conv1d/Squeeze:output:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_1/BiasAddl
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_2/conv1d/ExpandDims�
'fc_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
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
:  2
fc_2/conv1d/ExpandDims_1�
fc_2/conv1dConv2Dfc_2/conv1d/ExpandDims:output:0!fc_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingVALID*
strides
2
fc_2/conv1d�
fc_2/conv1d/SqueezeSqueezefc_2/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
fc_2/conv1d/Squeeze�
fc_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_2/BiasAdd/ReadVariableOp�
fc_2/BiasAddBiasAddfc_2/conv1d/Squeeze:output:0#fc_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
fc_2/BiasAddl
	fc_2/ReluRelufc_2/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
	fc_2/Reluz
fc_3/IdentityIdentityfc_2/Relu:activations:0*
T0*,
_output_shapes
:���������� 2
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
:���������� 2
fc_4/ExpandDims�
fc_4/MaxPoolMaxPoolfc_4/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingVALID*
strides
2
fc_4/MaxPool�
fc_4/SqueezeSqueezefc_4/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
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
:���������� 2
fc_5/conv1d/ExpandDims�
'fc_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
fc_5/conv1d/ExpandDims_1�
fc_5/conv1dConv2Dfc_5/conv1d/ExpandDims:output:0!fc_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_5/conv1d�
fc_5/conv1d/SqueezeSqueezefc_5/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_5/conv1d/Squeeze�
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_5/BiasAdd/ReadVariableOp�
fc_5/BiasAddBiasAddfc_5/conv1d/Squeeze:output:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_5/BiasAddl
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
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
:����������2
fc_6/conv1d/ExpandDims�
'fc_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp0fc_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
fc_6/conv1d/ExpandDims_1�
fc_6/conv1dConv2Dfc_6/conv1d/ExpandDims:output:0!fc_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
fc_6/conv1d�
fc_6/conv1d/SqueezeSqueezefc_6/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
fc_6/conv1d/Squeeze�
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_6/BiasAdd/ReadVariableOp�
fc_6/BiasAddBiasAddfc_6/conv1d/Squeeze:output:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
fc_6/BiasAddl
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
	fc_6/Reluz
fc_7/IdentityIdentityfc_6/Relu:activations:0*
T0*,
_output_shapes
:����������2
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
:����������2
fc_8/ExpandDims�
fc_8/MaxPoolMaxPoolfc_8/ExpandDims:output:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
fc_8/MaxPool�
fc_8/SqueezeSqueezefc_8/MaxPool:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims
2
fc_8/Squeezeg
	fc9/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
	fc9/Const�
fc9/ReshapeReshapefc_8/Squeeze:output:0fc9/Const:output:0*
T0*(
_output_shapes
:����������2
fc9/Reshape�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMulfc9/Reshape:output:0$output/MatMul/ReadVariableOp:value:0*
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
: *
dtype02/
-fc_1/kernel/Regularizer/Square/ReadVariableOp�
fc_1/kernel/Regularizer/SquareSquare5fc_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:  *
dtype02/
-fc_2/kernel/Regularizer/Square/ReadVariableOp�
fc_2/kernel/Regularizer/SquareSquare5fc_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:  2 
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
: *
dtype02/
-fc_5/kernel/Regularizer/Square/ReadVariableOp�
fc_5/kernel/Regularizer/SquareSquare5fc_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2 
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
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
fc_6/kernel/Regularizer/mul�
IdentityIdentityoutput/Softmax:softmax:0^fc_1/BiasAdd/ReadVariableOp(^fc_1/conv1d/ExpandDims_1/ReadVariableOp.^fc_1/kernel/Regularizer/Square/ReadVariableOp^fc_2/BiasAdd/ReadVariableOp(^fc_2/conv1d/ExpandDims_1/ReadVariableOp.^fc_2/kernel/Regularizer/Square/ReadVariableOp^fc_5/BiasAdd/ReadVariableOp(^fc_5/conv1d/ExpandDims_1/ReadVariableOp.^fc_5/kernel/Regularizer/Square/ReadVariableOp^fc_6/BiasAdd/ReadVariableOp(^fc_6/conv1d/ExpandDims_1/ReadVariableOp.^fc_6/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp2R
'fc_1/conv1d/ExpandDims_1/ReadVariableOp'fc_1/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_1/kernel/Regularizer/Square/ReadVariableOp-fc_1/kernel/Regularizer/Square/ReadVariableOp2:
fc_2/BiasAdd/ReadVariableOpfc_2/BiasAdd/ReadVariableOp2R
'fc_2/conv1d/ExpandDims_1/ReadVariableOp'fc_2/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_2/kernel/Regularizer/Square/ReadVariableOp-fc_2/kernel/Regularizer/Square/ReadVariableOp2:
fc_5/BiasAdd/ReadVariableOpfc_5/BiasAdd/ReadVariableOp2R
'fc_5/conv1d/ExpandDims_1/ReadVariableOp'fc_5/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_5/kernel/Regularizer/Square/ReadVariableOp-fc_5/kernel/Regularizer/Square/ReadVariableOp2:
fc_6/BiasAdd/ReadVariableOpfc_6/BiasAdd/ReadVariableOp2R
'fc_6/conv1d/ExpandDims_1/ReadVariableOp'fc_6/conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_fc_8_layer_call_and_return_conditional_losses_2759072

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
C__inference_output_layer_call_and_return_conditional_losses_2759417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
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
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_fc_6_layer_call_and_return_conditional_losses_2758833

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
:����������2
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
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
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
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
-fc_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-fc_6/kernel/Regularizer/Square/ReadVariableOp�
fc_6/kernel/Regularizer/SquareSquare5fc_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:2 
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
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2^
-fc_6/kernel/Regularizer/Square/ReadVariableOp-fc_6/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_mnist_layer_call_fn_2759723

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
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_27597082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
@
%__inference_signature_wrapper_2760358	
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
__inference_pruned_27603512
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
�
_
A__inference_fc_3_layer_call_and_return_conditional_losses_2759143

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
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
tensorflow/serving/predict:��
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
layer-11
layer_with_weights-4
layer-12
regularization_losses
trainable_variables
	variables
	keras_api

signatures
#_self_saveable_object_factories
trt_engine_resources
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"
_generic_user_object
C
#_self_saveable_object_factories"
_generic_user_object
�
regularization_losses
trainable_variables
	variables
	keras_api
#_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
R
	keras_api
#_self_saveable_object_factories"
_generic_user_object
�

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
##_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
#*_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
+regularization_losses
,trainable_variables
-	variables
.	keras_api
#/_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
0regularization_losses
1trainable_variables
2	variables
3	keras_api
#4_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
#;_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
#B_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
#G_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
#L_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
#Q_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
�

Rkernel
Sbias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
#X_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses"
_generic_user_object
@
�0
�1
�2
�3"
trackable_list_wrapper
f
0
1
$2
%3
54
65
<6
=7
R8
S9"
trackable_list_wrapper
f
0
1
$2
%3
54
65
<6
=7
R8
S9"
trackable_list_wrapper
�
Ynon_trainable_variables
regularization_losses
Zlayer_metrics
[metrics
\layer_regularization_losses
trainable_variables

]layers
	variables
#^_self_saveable_object_factories
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
_non_trainable_variables
regularization_losses
`layer_metrics
ametrics
blayer_regularization_losses
trainable_variables

clayers
	variables
#d_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
C
#e_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
!: 2fc_1/kernel
: 2	fc_1/bias
(
�0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
fnon_trainable_variables
regularization_losses
glayer_metrics
hmetrics
ilayer_regularization_losses
 trainable_variables

jlayers
!	variables
#k_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!:  2fc_2/kernel
: 2	fc_2/bias
(
�0"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
lnon_trainable_variables
&regularization_losses
mlayer_metrics
nmetrics
olayer_regularization_losses
'trainable_variables

players
(	variables
#q_self_saveable_object_factories
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
rnon_trainable_variables
+regularization_losses
slayer_metrics
tmetrics
ulayer_regularization_losses
,trainable_variables

vlayers
-	variables
#w_self_saveable_object_factories
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
xnon_trainable_variables
0regularization_losses
ylayer_metrics
zmetrics
{layer_regularization_losses
1trainable_variables

|layers
2	variables
#}_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!: 2fc_5/kernel
:2	fc_5/bias
(
�0"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
~non_trainable_variables
7regularization_losses
layer_metrics
�metrics
 �layer_regularization_losses
8trainable_variables
�layers
9	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
!:2fc_6/kernel
:2	fc_6/bias
(
�0"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
�
�non_trainable_variables
>regularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
?trainable_variables
�layers
@	variables
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
Cregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Dtrainable_variables
�layers
E	variables
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
Hregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Itrainable_variables
�layers
J	variables
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
Mregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Ntrainable_variables
�layers
O	variables
$�_self_saveable_object_factories
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 :	�
2output/kernel
:
2output/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
�
�non_trainable_variables
Tregularization_losses
�layer_metrics
�metrics
 �layer_regularization_losses
Utrainable_variables
�layers
V	variables
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
12"
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
'__inference_mnist_layer_call_fn_2759723
'__inference_mnist_layer_call_fn_2759738
'__inference_mnist_layer_call_fn_2759555
'__inference_mnist_layer_call_fn_2759540�
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
"__inference__wrapped_model_2758995�
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
B__inference_mnist_layer_call_and_return_conditional_losses_2759400
B__inference_mnist_layer_call_and_return_conditional_losses_2759657
B__inference_mnist_layer_call_and_return_conditional_losses_2759289
B__inference_mnist_layer_call_and_return_conditional_losses_2759606�
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
)__inference_flatten_layer_call_fn_2758773�
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
D__inference_flatten_layer_call_and_return_conditional_losses_2758897�
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
&__inference_fc_1_layer_call_fn_2758891�
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
A__inference_fc_1_layer_call_and_return_conditional_losses_2759099�
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
&__inference_fc_2_layer_call_fn_2758862�
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
A__inference_fc_2_layer_call_and_return_conditional_losses_2759138�
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
&__inference_fc_3_layer_call_fn_2758735
&__inference_fc_3_layer_call_fn_2758907�
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
A__inference_fc_3_layer_call_and_return_conditional_losses_2759143
A__inference_fc_3_layer_call_and_return_conditional_losses_2759770�
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
&__inference_fc_4_layer_call_fn_2758920�
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
A__inference_fc_4_layer_call_and_return_conditional_losses_2758915�
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
&__inference_fc_5_layer_call_fn_2759039�
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
A__inference_fc_5_layer_call_and_return_conditional_losses_2759176�
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
&__inference_fc_6_layer_call_fn_2759474�
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
A__inference_fc_6_layer_call_and_return_conditional_losses_2758833�
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
&__inference_fc_7_layer_call_fn_2759427
&__inference_fc_7_layer_call_fn_2759116�
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
A__inference_fc_7_layer_call_and_return_conditional_losses_2759054
A__inference_fc_7_layer_call_and_return_conditional_losses_2759301�
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
&__inference_fc_8_layer_call_fn_2759077�
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
A__inference_fc_8_layer_call_and_return_conditional_losses_2759072�
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
%__inference_fc9_layer_call_fn_2759154�
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
@__inference_fc9_layer_call_and_return_conditional_losses_2759406�
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
(__inference_output_layer_call_fn_2759445�
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
C__inference_output_layer_call_and_return_conditional_losses_2759417�
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
__inference_loss_fn_0_2759049�
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
__inference_loss_fn_1_2759758�
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
__inference_loss_fn_2_2759748�
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
__inference_loss_fn_3_2759064�
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
%__inference_signature_wrapper_2760358input"�
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
"__inference__wrapped_model_2758995u
$%56<=RS6�3
,�)
'�$
input���������
� "/�,
*
output �
output���������
�
@__inference_fc9_layer_call_and_return_conditional_losses_2759406^4�1
*�'
%�"
inputs����������
� "&�#
�
0����������
� z
%__inference_fc9_layer_call_fn_2759154Q4�1
*�'
%�"
inputs����������
� "������������
A__inference_fc_1_layer_call_and_return_conditional_losses_2759099f4�1
*�'
%�"
inputs����������
� "*�'
 �
0���������� 
� �
&__inference_fc_1_layer_call_fn_2758891Y4�1
*�'
%�"
inputs����������
� "����������� �
A__inference_fc_2_layer_call_and_return_conditional_losses_2759138f$%4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0���������� 
� �
&__inference_fc_2_layer_call_fn_2758862Y$%4�1
*�'
%�"
inputs���������� 
� "����������� �
A__inference_fc_3_layer_call_and_return_conditional_losses_2759143f8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
A__inference_fc_3_layer_call_and_return_conditional_losses_2759770f8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
&__inference_fc_3_layer_call_fn_2758735Y8�5
.�+
%�"
inputs���������� 
p
� "����������� �
&__inference_fc_3_layer_call_fn_2758907Y8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
A__inference_fc_4_layer_call_and_return_conditional_losses_2758915�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_fc_4_layer_call_fn_2758920wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
A__inference_fc_5_layer_call_and_return_conditional_losses_2759176f564�1
*�'
%�"
inputs���������� 
� "*�'
 �
0����������
� �
&__inference_fc_5_layer_call_fn_2759039Y564�1
*�'
%�"
inputs���������� 
� "������������
A__inference_fc_6_layer_call_and_return_conditional_losses_2758833f<=4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
&__inference_fc_6_layer_call_fn_2759474Y<=4�1
*�'
%�"
inputs����������
� "������������
A__inference_fc_7_layer_call_and_return_conditional_losses_2759054f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
A__inference_fc_7_layer_call_and_return_conditional_losses_2759301f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
&__inference_fc_7_layer_call_fn_2759116Y8�5
.�+
%�"
inputs����������
p
� "������������
&__inference_fc_7_layer_call_fn_2759427Y8�5
.�+
%�"
inputs����������
p 
� "������������
A__inference_fc_8_layer_call_and_return_conditional_losses_2759072�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_fc_8_layer_call_fn_2759077wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
D__inference_flatten_layer_call_and_return_conditional_losses_2758897a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
)__inference_flatten_layer_call_fn_2758773T7�4
-�*
(�%
inputs���������
� "�����������<
__inference_loss_fn_0_2759049�

� 
� "� <
__inference_loss_fn_1_2759758$�

� 
� "� <
__inference_loss_fn_2_27597485�

� 
� "� <
__inference_loss_fn_3_2759064<�

� 
� "� �
B__inference_mnist_layer_call_and_return_conditional_losses_2759289t
$%56<=RS?�<
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
B__inference_mnist_layer_call_and_return_conditional_losses_2759400t
$%56<=RS?�<
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
B__inference_mnist_layer_call_and_return_conditional_losses_2759606s
$%56<=RS>�;
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
B__inference_mnist_layer_call_and_return_conditional_losses_2759657s
$%56<=RS>�;
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
'__inference_mnist_layer_call_fn_2759540f
$%56<=RS>�;
4�1
'�$
input���������
p

 
� "����������
�
'__inference_mnist_layer_call_fn_2759555g
$%56<=RS?�<
5�2
(�%
inputs���������
p

 
� "����������
�
'__inference_mnist_layer_call_fn_2759723g
$%56<=RS?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
'__inference_mnist_layer_call_fn_2759738f
$%56<=RS>�;
4�1
'�$
input���������
p 

 
� "����������
�
C__inference_output_layer_call_and_return_conditional_losses_2759417]RS0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� |
(__inference_output_layer_call_fn_2759445PRS0�-
&�#
!�
inputs����������
� "����������
�
%__inference_signature_wrapper_2760358r?�<
� 
5�2
0
input'�$
input���������"/�,
*
output �
output���������
