ср
ЈЫ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Њ
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12unknown8∞Ё	
t
fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
РА*
shared_namefc_1/kernel
m
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel* 
_output_shapes
:
РА*
dtype0
k
	fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	fc_1/bias
d
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes	
:А*
dtype0
t
fc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namefc_2/kernel
m
fc_2/kernel/Read/ReadVariableOpReadVariableOpfc_2/kernel* 
_output_shapes
:
АА*
dtype0
k
	fc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	fc_2/bias
d
fc_2/bias/Read/ReadVariableOpReadVariableOp	fc_2/bias*
_output_shapes	
:А*
dtype0
t
fc_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namefc_3/kernel
m
fc_3/kernel/Read/ReadVariableOpReadVariableOpfc_3/kernel* 
_output_shapes
:
АА*
dtype0
k
	fc_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	fc_3/bias
d
fc_3/bias/Read/ReadVariableOpReadVariableOp	fc_3/bias*
_output_shapes	
:А*
dtype0
t
fc_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namefc_4/kernel
m
fc_4/kernel/Read/ReadVariableOpReadVariableOpfc_4/kernel* 
_output_shapes
:
АА*
dtype0
k
	fc_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	fc_4/bias
d
fc_4/bias/Read/ReadVariableOpReadVariableOp	fc_4/bias*
_output_shapes	
:А*
dtype0
t
fc_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namefc_5/kernel
m
fc_5/kernel/Read/ReadVariableOpReadVariableOpfc_5/kernel* 
_output_shapes
:
АА*
dtype0
k
	fc_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	fc_5/bias
d
fc_5/bias/Read/ReadVariableOpReadVariableOp	fc_5/bias*
_output_shapes	
:А*
dtype0
t
fc_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namefc_6/kernel
m
fc_6/kernel/Read/ReadVariableOpReadVariableOpfc_6/kernel* 
_output_shapes
:
АА*
dtype0
k
	fc_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	fc_6/bias
d
fc_6/bias/Read/ReadVariableOpReadVariableOp	fc_6/bias*
_output_shapes	
:А*
dtype0
s
fc_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namefc_7/kernel
l
fc_7/kernel/Read/ReadVariableOpReadVariableOpfc_7/kernel*
_output_shapes
:	А@*
dtype0
j
	fc_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_7/bias
c
fc_7/bias/Read/ReadVariableOpReadVariableOp	fc_7/bias*
_output_shapes
:@*
dtype0
r
fc_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namefc_8/kernel
k
fc_8/kernel/Read/ReadVariableOpReadVariableOpfc_8/kernel*
_output_shapes

:@@*
dtype0
j
	fc_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_8/bias
c
fc_8/bias/Read/ReadVariableOpReadVariableOp	fc_8/bias*
_output_shapes
:@*
dtype0
r
fc_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namefc_9/kernel
k
fc_9/kernel/Read/ReadVariableOpReadVariableOpfc_9/kernel*
_output_shapes

:@ *
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
t
fc_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namefc_10/kernel
m
 fc_10/kernel/Read/ReadVariableOpReadVariableOpfc_10/kernel*
_output_shapes

:  *
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
t
fc_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namefc_11/kernel
m
 fc_11/kernel/Read/ReadVariableOpReadVariableOpfc_11/kernel*
_output_shapes

: *
dtype0
l

fc_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc_11/bias
e
fc_11/bias/Read/ReadVariableOpReadVariableOp
fc_11/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:
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
А<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ї;
value±;BЃ; BІ;
÷
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
h

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
h

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
ґ
0
1
2
3
$4
%5
*6
+7
08
19
610
711
<12
=13
B14
C15
H16
I17
N18
O19
T20
U21
Z22
[23
ґ
0
1
2
3
$4
%5
*6
+7
08
19
610
711
<12
=13
B14
C15
H16
I17
N18
O19
T20
U21
Z22
[23
 
≠
`layer_metrics
	variables
anon_trainable_variables
bmetrics

clayers
dlayer_regularization_losses
trainable_variables
regularization_losses
 
 
 
 
≠
elayer_metrics
	variables
fmetrics
gnon_trainable_variables

hlayers
ilayer_regularization_losses
trainable_variables
regularization_losses
WU
VARIABLE_VALUEfc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
jlayer_metrics
	variables
kmetrics
lnon_trainable_variables

mlayers
nlayer_regularization_losses
trainable_variables
regularization_losses
WU
VARIABLE_VALUEfc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
olayer_metrics
 	variables
pmetrics
qnon_trainable_variables

rlayers
slayer_regularization_losses
!trainable_variables
"regularization_losses
WU
VARIABLE_VALUEfc_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
≠
tlayer_metrics
&	variables
umetrics
vnon_trainable_variables

wlayers
xlayer_regularization_losses
'trainable_variables
(regularization_losses
WU
VARIABLE_VALUEfc_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
≠
ylayer_metrics
,	variables
zmetrics
{non_trainable_variables

|layers
}layer_regularization_losses
-trainable_variables
.regularization_losses
WU
VARIABLE_VALUEfc_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
∞
~layer_metrics
2	variables
metrics
Аnon_trainable_variables
Бlayers
 Вlayer_regularization_losses
3trainable_variables
4regularization_losses
WU
VARIABLE_VALUEfc_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
≤
Гlayer_metrics
8	variables
Дmetrics
Еnon_trainable_variables
Жlayers
 Зlayer_regularization_losses
9trainable_variables
:regularization_losses
WU
VARIABLE_VALUEfc_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
≤
Иlayer_metrics
>	variables
Йmetrics
Кnon_trainable_variables
Лlayers
 Мlayer_regularization_losses
?trainable_variables
@regularization_losses
WU
VARIABLE_VALUEfc_8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
≤
Нlayer_metrics
D	variables
Оmetrics
Пnon_trainable_variables
Рlayers
 Сlayer_regularization_losses
Etrainable_variables
Fregularization_losses
WU
VARIABLE_VALUEfc_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
≤
Тlayer_metrics
J	variables
Уmetrics
Фnon_trainable_variables
Хlayers
 Цlayer_regularization_losses
Ktrainable_variables
Lregularization_losses
XV
VARIABLE_VALUEfc_10/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
fc_10/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 
≤
Чlayer_metrics
P	variables
Шmetrics
Щnon_trainable_variables
Ъlayers
 Ыlayer_regularization_losses
Qtrainable_variables
Rregularization_losses
YW
VARIABLE_VALUEfc_11/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
fc_11/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
≤
Ьlayer_metrics
V	variables
Эmetrics
Юnon_trainable_variables
Яlayers
 †layer_regularization_losses
Wtrainable_variables
Xregularization_losses
ZX
VARIABLE_VALUEoutput/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEoutput/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

Z0
[1
 
≤
°layer_metrics
\	variables
Ґmetrics
£non_trainable_variables
§layers
 •layer_regularization_losses
]trainable_variables
^regularization_losses
 
 
 
f
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
И
serving_default_inputPlaceholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
М
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputfc_1/kernel	fc_1/biasfc_2/kernel	fc_2/biasfc_3/kernel	fc_3/biasfc_4/kernel	fc_4/biasfc_5/kernel	fc_5/biasfc_6/kernel	fc_6/biasfc_7/kernel	fc_7/biasfc_8/kernel	fc_8/biasfc_9/kernel	fc_9/biasfc_10/kernel
fc_10/biasfc_11/kernel
fc_11/biasoutput/kerneloutput/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_6304711
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
њ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOpfc_2/kernel/Read/ReadVariableOpfc_2/bias/Read/ReadVariableOpfc_3/kernel/Read/ReadVariableOpfc_3/bias/Read/ReadVariableOpfc_4/kernel/Read/ReadVariableOpfc_4/bias/Read/ReadVariableOpfc_5/kernel/Read/ReadVariableOpfc_5/bias/Read/ReadVariableOpfc_6/kernel/Read/ReadVariableOpfc_6/bias/Read/ReadVariableOpfc_7/kernel/Read/ReadVariableOpfc_7/bias/Read/ReadVariableOpfc_8/kernel/Read/ReadVariableOpfc_8/bias/Read/ReadVariableOpfc_9/kernel/Read/ReadVariableOpfc_9/bias/Read/ReadVariableOp fc_10/kernel/Read/ReadVariableOpfc_10/bias/Read/ReadVariableOp fc_11/kernel/Read/ReadVariableOpfc_11/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpConst*%
Tin
2*
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_6305343
Џ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefc_1/kernel	fc_1/biasfc_2/kernel	fc_2/biasfc_3/kernel	fc_3/biasfc_4/kernel	fc_4/biasfc_5/kernel	fc_5/biasfc_6/kernel	fc_6/biasfc_7/kernel	fc_7/biasfc_8/kernel	fc_8/biasfc_9/kernel	fc_9/biasfc_10/kernel
fc_10/biasfc_11/kernel
fc_11/biasoutput/kerneloutput/bias*$
Tin
2*
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_6305425мг
ƒ
“
'__inference_mnist_layer_call_fn_6304656	
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_63046052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameinput
ф	
Џ
A__inference_fc_4_layer_call_and_return_conditional_losses_6304121

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
х	
№
C__inference_output_layer_call_and_return_conditional_losses_6304337

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ў
{
&__inference_fc_8_layer_call_fn_6305168

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_63042292
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_6_layer_call_and_return_conditional_losses_6305119

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
л	
Џ
A__inference_fc_9_layer_call_and_return_conditional_losses_6305179

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
о	
Џ
A__inference_fc_7_layer_call_and_return_conditional_losses_6304202

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
–=
Р
B__inference_mnist_layer_call_and_return_conditional_losses_6304419	
input
fc_1_6304358
fc_1_6304360
fc_2_6304363
fc_2_6304365
fc_3_6304368
fc_3_6304370
fc_4_6304373
fc_4_6304375
fc_5_6304378
fc_5_6304380
fc_6_6304383
fc_6_6304385
fc_7_6304388
fc_7_6304390
fc_8_6304393
fc_8_6304395
fc_9_6304398
fc_9_6304400
fc_10_6304403
fc_10_6304405
fc_11_6304408
fc_11_6304410
output_6304413
output_6304415
identityИҐfc_1/StatefulPartitionedCallҐfc_10/StatefulPartitionedCallҐfc_11/StatefulPartitionedCallҐfc_2/StatefulPartitionedCallҐfc_3/StatefulPartitionedCallҐfc_4/StatefulPartitionedCallҐfc_5/StatefulPartitionedCallҐfc_6/StatefulPartitionedCallҐfc_7/StatefulPartitionedCallҐfc_8/StatefulPartitionedCallҐfc_9/StatefulPartitionedCallҐoutput/StatefulPartitionedCall’
flatten/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_63040212
flatten/PartitionedCall°
fc_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1_6304358fc_1_6304360*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_63040402
fc_1/StatefulPartitionedCall¶
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_6304363fc_2_6304365*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_63040672
fc_2/StatefulPartitionedCall¶
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0fc_3_6304368fc_3_6304370*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_63040942
fc_3/StatefulPartitionedCall¶
fc_4/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0fc_4_6304373fc_4_6304375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_63041212
fc_4/StatefulPartitionedCall¶
fc_5/StatefulPartitionedCallStatefulPartitionedCall%fc_4/StatefulPartitionedCall:output:0fc_5_6304378fc_5_6304380*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_63041482
fc_5/StatefulPartitionedCall¶
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_6304383fc_6_6304385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_63041752
fc_6/StatefulPartitionedCall•
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0fc_7_6304388fc_7_6304390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_63042022
fc_7/StatefulPartitionedCall•
fc_8/StatefulPartitionedCallStatefulPartitionedCall%fc_7/StatefulPartitionedCall:output:0fc_8_6304393fc_8_6304395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_63042292
fc_8/StatefulPartitionedCall•
fc_9/StatefulPartitionedCallStatefulPartitionedCall%fc_8/StatefulPartitionedCall:output:0fc_9_6304398fc_9_6304400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_63042562
fc_9/StatefulPartitionedCall™
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_6304403fc_10_6304405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_63042832
fc_10/StatefulPartitionedCallЂ
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0fc_11_6304408fc_11_6304410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_63043102
fc_11/StatefulPartitionedCall∞
output/StatefulPartitionedCallStatefulPartitionedCall&fc_11/StatefulPartitionedCall:output:0output_6304413output_6304415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_63043372 
output/StatefulPartitionedCallу
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall^fc_10/StatefulPartitionedCall^fc_11/StatefulPartitionedCall^fc_2/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^fc_4/StatefulPartitionedCall^fc_5/StatefulPartitionedCall^fc_6/StatefulPartitionedCall^fc_7/StatefulPartitionedCall^fc_8/StatefulPartitionedCall^fc_9/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2>
fc_11/StatefulPartitionedCallfc_11/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_4/StatefulPartitionedCallfc_4/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2<
fc_8/StatefulPartitionedCallfc_8/StatefulPartitionedCall2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameinput
о	
Џ
A__inference_fc_7_layer_call_and_return_conditional_losses_6305139

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ё
{
&__inference_fc_2_layer_call_fn_6305048

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_63040672
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_2_layer_call_and_return_conditional_losses_6304067

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_6_layer_call_and_return_conditional_losses_6304175

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
”=
С
B__inference_mnist_layer_call_and_return_conditional_losses_6304605

inputs
fc_1_6304544
fc_1_6304546
fc_2_6304549
fc_2_6304551
fc_3_6304554
fc_3_6304556
fc_4_6304559
fc_4_6304561
fc_5_6304564
fc_5_6304566
fc_6_6304569
fc_6_6304571
fc_7_6304574
fc_7_6304576
fc_8_6304579
fc_8_6304581
fc_9_6304584
fc_9_6304586
fc_10_6304589
fc_10_6304591
fc_11_6304594
fc_11_6304596
output_6304599
output_6304601
identityИҐfc_1/StatefulPartitionedCallҐfc_10/StatefulPartitionedCallҐfc_11/StatefulPartitionedCallҐfc_2/StatefulPartitionedCallҐfc_3/StatefulPartitionedCallҐfc_4/StatefulPartitionedCallҐfc_5/StatefulPartitionedCallҐfc_6/StatefulPartitionedCallҐfc_7/StatefulPartitionedCallҐfc_8/StatefulPartitionedCallҐfc_9/StatefulPartitionedCallҐoutput/StatefulPartitionedCall÷
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_63040212
flatten/PartitionedCall°
fc_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1_6304544fc_1_6304546*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_63040402
fc_1/StatefulPartitionedCall¶
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_6304549fc_2_6304551*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_63040672
fc_2/StatefulPartitionedCall¶
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0fc_3_6304554fc_3_6304556*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_63040942
fc_3/StatefulPartitionedCall¶
fc_4/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0fc_4_6304559fc_4_6304561*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_63041212
fc_4/StatefulPartitionedCall¶
fc_5/StatefulPartitionedCallStatefulPartitionedCall%fc_4/StatefulPartitionedCall:output:0fc_5_6304564fc_5_6304566*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_63041482
fc_5/StatefulPartitionedCall¶
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_6304569fc_6_6304571*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_63041752
fc_6/StatefulPartitionedCall•
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0fc_7_6304574fc_7_6304576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_63042022
fc_7/StatefulPartitionedCall•
fc_8/StatefulPartitionedCallStatefulPartitionedCall%fc_7/StatefulPartitionedCall:output:0fc_8_6304579fc_8_6304581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_63042292
fc_8/StatefulPartitionedCall•
fc_9/StatefulPartitionedCallStatefulPartitionedCall%fc_8/StatefulPartitionedCall:output:0fc_9_6304584fc_9_6304586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_63042562
fc_9/StatefulPartitionedCall™
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_6304589fc_10_6304591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_63042832
fc_10/StatefulPartitionedCallЂ
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0fc_11_6304594fc_11_6304596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_63043102
fc_11/StatefulPartitionedCall∞
output/StatefulPartitionedCallStatefulPartitionedCall&fc_11/StatefulPartitionedCall:output:0output_6304599output_6304601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_63043372 
output/StatefulPartitionedCallу
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall^fc_10/StatefulPartitionedCall^fc_11/StatefulPartitionedCall^fc_2/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^fc_4/StatefulPartitionedCall^fc_5/StatefulPartitionedCall^fc_6/StatefulPartitionedCall^fc_7/StatefulPartitionedCall^fc_8/StatefulPartitionedCall^fc_9/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2>
fc_11/StatefulPartitionedCallfc_11/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_4/StatefulPartitionedCallfc_4/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2<
fc_8/StatefulPartitionedCallfc_8/StatefulPartitionedCall2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Љ
`
D__inference_flatten_layer_call_and_return_conditional_losses_6304021

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_3_layer_call_and_return_conditional_losses_6304094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ґ
–
%__inference_signature_wrapper_6304711	
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_63040112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameinput
м	
џ
B__inference_fc_11_layer_call_and_return_conditional_losses_6304310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џ
|
'__inference_fc_11_layer_call_fn_6305228

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_63043102
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џ
|
'__inference_fc_10_layer_call_fn_6305208

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_63042832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џ
{
&__inference_fc_7_layer_call_fn_6305148

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_63042022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
м	
џ
B__inference_fc_10_layer_call_and_return_conditional_losses_6304283

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_5_layer_call_and_return_conditional_losses_6304148

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ё
{
&__inference_fc_4_layer_call_fn_6305088

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_63041212
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
л	
Џ
A__inference_fc_9_layer_call_and_return_conditional_losses_6304256

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
«
”
'__inference_mnist_layer_call_fn_6304997

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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_63046052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
{
&__inference_fc_1_layer_call_fn_6305028

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_63040402
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
Љ
`
D__inference_flatten_layer_call_and_return_conditional_losses_6305003

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
{
&__inference_fc_3_layer_call_fn_6305068

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_63040942
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ќh
Щ
B__inference_mnist_layer_call_and_return_conditional_losses_6304801

inputs'
#fc_1_matmul_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource'
#fc_2_matmul_readvariableop_resource(
$fc_2_biasadd_readvariableop_resource'
#fc_3_matmul_readvariableop_resource(
$fc_3_biasadd_readvariableop_resource'
#fc_4_matmul_readvariableop_resource(
$fc_4_biasadd_readvariableop_resource'
#fc_5_matmul_readvariableop_resource(
$fc_5_biasadd_readvariableop_resource'
#fc_6_matmul_readvariableop_resource(
$fc_6_biasadd_readvariableop_resource'
#fc_7_matmul_readvariableop_resource(
$fc_7_biasadd_readvariableop_resource'
#fc_8_matmul_readvariableop_resource(
$fc_8_biasadd_readvariableop_resource'
#fc_9_matmul_readvariableop_resource(
$fc_9_biasadd_readvariableop_resource(
$fc_10_matmul_readvariableop_resource)
%fc_10_biasadd_readvariableop_resource(
$fc_11_matmul_readvariableop_resource)
%fc_11_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityИҐfc_1/BiasAdd/ReadVariableOpҐfc_1/MatMul/ReadVariableOpҐfc_10/BiasAdd/ReadVariableOpҐfc_10/MatMul/ReadVariableOpҐfc_11/BiasAdd/ReadVariableOpҐfc_11/MatMul/ReadVariableOpҐfc_2/BiasAdd/ReadVariableOpҐfc_2/MatMul/ReadVariableOpҐfc_3/BiasAdd/ReadVariableOpҐfc_3/MatMul/ReadVariableOpҐfc_4/BiasAdd/ReadVariableOpҐfc_4/MatMul/ReadVariableOpҐfc_5/BiasAdd/ReadVariableOpҐfc_5/MatMul/ReadVariableOpҐfc_6/BiasAdd/ReadVariableOpҐfc_6/MatMul/ReadVariableOpҐfc_7/BiasAdd/ReadVariableOpҐfc_7/MatMul/ReadVariableOpҐfc_8/BiasAdd/ReadVariableOpҐfc_8/MatMul/ReadVariableOpҐfc_9/BiasAdd/ReadVariableOpҐfc_9/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
flatten/ConstА
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
flatten/ReshapeЮ
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource* 
_output_shapes
:
РА*
dtype02
fc_1/MatMul/ReadVariableOpХ
fc_1/MatMulMatMulflatten/Reshape:output:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_1/MatMulЬ
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_1/BiasAdd/ReadVariableOpЦ
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_1/BiasAddh
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_1/ReluЮ
fc_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_2/MatMul/ReadVariableOpФ
fc_2/MatMulMatMulfc_1/Relu:activations:0"fc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_2/MatMulЬ
fc_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_2/BiasAdd/ReadVariableOpЦ
fc_2/BiasAddBiasAddfc_2/MatMul:product:0#fc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_2/BiasAddh
	fc_2/ReluRelufc_2/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_2/ReluЮ
fc_3/MatMul/ReadVariableOpReadVariableOp#fc_3_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_3/MatMul/ReadVariableOpФ
fc_3/MatMulMatMulfc_2/Relu:activations:0"fc_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_3/MatMulЬ
fc_3/BiasAdd/ReadVariableOpReadVariableOp$fc_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_3/BiasAdd/ReadVariableOpЦ
fc_3/BiasAddBiasAddfc_3/MatMul:product:0#fc_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_3/BiasAddh
	fc_3/ReluRelufc_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_3/ReluЮ
fc_4/MatMul/ReadVariableOpReadVariableOp#fc_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_4/MatMul/ReadVariableOpФ
fc_4/MatMulMatMulfc_3/Relu:activations:0"fc_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_4/MatMulЬ
fc_4/BiasAdd/ReadVariableOpReadVariableOp$fc_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_4/BiasAdd/ReadVariableOpЦ
fc_4/BiasAddBiasAddfc_4/MatMul:product:0#fc_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_4/BiasAddh
	fc_4/ReluRelufc_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_4/ReluЮ
fc_5/MatMul/ReadVariableOpReadVariableOp#fc_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_5/MatMul/ReadVariableOpФ
fc_5/MatMulMatMulfc_4/Relu:activations:0"fc_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_5/MatMulЬ
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_5/BiasAdd/ReadVariableOpЦ
fc_5/BiasAddBiasAddfc_5/MatMul:product:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_5/BiasAddh
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_5/ReluЮ
fc_6/MatMul/ReadVariableOpReadVariableOp#fc_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_6/MatMul/ReadVariableOpФ
fc_6/MatMulMatMulfc_5/Relu:activations:0"fc_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_6/MatMulЬ
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_6/BiasAdd/ReadVariableOpЦ
fc_6/BiasAddBiasAddfc_6/MatMul:product:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_6/BiasAddh
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_6/ReluЭ
fc_7/MatMul/ReadVariableOpReadVariableOp#fc_7_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
fc_7/MatMul/ReadVariableOpУ
fc_7/MatMulMatMulfc_6/Relu:activations:0"fc_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_7/MatMulЫ
fc_7/BiasAdd/ReadVariableOpReadVariableOp$fc_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_7/BiasAdd/ReadVariableOpХ
fc_7/BiasAddBiasAddfc_7/MatMul:product:0#fc_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_7/BiasAddg
	fc_7/ReluRelufc_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	fc_7/ReluЬ
fc_8/MatMul/ReadVariableOpReadVariableOp#fc_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
fc_8/MatMul/ReadVariableOpУ
fc_8/MatMulMatMulfc_7/Relu:activations:0"fc_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_8/MatMulЫ
fc_8/BiasAdd/ReadVariableOpReadVariableOp$fc_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_8/BiasAdd/ReadVariableOpХ
fc_8/BiasAddBiasAddfc_8/MatMul:product:0#fc_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_8/BiasAddg
	fc_8/ReluRelufc_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	fc_8/ReluЬ
fc_9/MatMul/ReadVariableOpReadVariableOp#fc_9_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
fc_9/MatMul/ReadVariableOpУ
fc_9/MatMulMatMulfc_8/Relu:activations:0"fc_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_9/MatMulЫ
fc_9/BiasAdd/ReadVariableOpReadVariableOp$fc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_9/BiasAdd/ReadVariableOpХ
fc_9/BiasAddBiasAddfc_9/MatMul:product:0#fc_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_9/BiasAddg
	fc_9/ReluRelufc_9/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	fc_9/ReluЯ
fc_10/MatMul/ReadVariableOpReadVariableOp$fc_10_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
fc_10/MatMul/ReadVariableOpЦ
fc_10/MatMulMatMulfc_9/Relu:activations:0#fc_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_10/MatMulЮ
fc_10/BiasAdd/ReadVariableOpReadVariableOp%fc_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_10/BiasAdd/ReadVariableOpЩ
fc_10/BiasAddBiasAddfc_10/MatMul:product:0$fc_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_10/BiasAddj

fc_10/ReluRelufc_10/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

fc_10/ReluЯ
fc_11/MatMul/ReadVariableOpReadVariableOp$fc_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
fc_11/MatMul/ReadVariableOpЧ
fc_11/MatMulMatMulfc_10/Relu:activations:0#fc_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
fc_11/MatMulЮ
fc_11/BiasAdd/ReadVariableOpReadVariableOp%fc_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_11/BiasAdd/ReadVariableOpЩ
fc_11/BiasAddBiasAddfc_11/MatMul:product:0$fc_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
fc_11/BiasAddj

fc_11/ReluRelufc_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

fc_11/ReluҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
output/MatMul/ReadVariableOpЪ
output/MatMulMatMulfc_11/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
output/SoftmaxЄ
IdentityIdentityoutput/Softmax:softmax:0^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^fc_10/BiasAdd/ReadVariableOp^fc_10/MatMul/ReadVariableOp^fc_11/BiasAdd/ReadVariableOp^fc_11/MatMul/ReadVariableOp^fc_2/BiasAdd/ReadVariableOp^fc_2/MatMul/ReadVariableOp^fc_3/BiasAdd/ReadVariableOp^fc_3/MatMul/ReadVariableOp^fc_4/BiasAdd/ReadVariableOp^fc_4/MatMul/ReadVariableOp^fc_5/BiasAdd/ReadVariableOp^fc_5/MatMul/ReadVariableOp^fc_6/BiasAdd/ReadVariableOp^fc_6/MatMul/ReadVariableOp^fc_7/BiasAdd/ReadVariableOp^fc_7/MatMul/ReadVariableOp^fc_8/BiasAdd/ReadVariableOp^fc_8/MatMul/ReadVariableOp^fc_9/BiasAdd/ReadVariableOp^fc_9/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2<
fc_10/BiasAdd/ReadVariableOpfc_10/BiasAdd/ReadVariableOp2:
fc_10/MatMul/ReadVariableOpfc_10/MatMul/ReadVariableOp2<
fc_11/BiasAdd/ReadVariableOpfc_11/BiasAdd/ReadVariableOp2:
fc_11/MatMul/ReadVariableOpfc_11/MatMul/ReadVariableOp2:
fc_2/BiasAdd/ReadVariableOpfc_2/BiasAdd/ReadVariableOp28
fc_2/MatMul/ReadVariableOpfc_2/MatMul/ReadVariableOp2:
fc_3/BiasAdd/ReadVariableOpfc_3/BiasAdd/ReadVariableOp28
fc_3/MatMul/ReadVariableOpfc_3/MatMul/ReadVariableOp2:
fc_4/BiasAdd/ReadVariableOpfc_4/BiasAdd/ReadVariableOp28
fc_4/MatMul/ReadVariableOpfc_4/MatMul/ReadVariableOp2:
fc_5/BiasAdd/ReadVariableOpfc_5/BiasAdd/ReadVariableOp28
fc_5/MatMul/ReadVariableOpfc_5/MatMul/ReadVariableOp2:
fc_6/BiasAdd/ReadVariableOpfc_6/BiasAdd/ReadVariableOp28
fc_6/MatMul/ReadVariableOpfc_6/MatMul/ReadVariableOp2:
fc_7/BiasAdd/ReadVariableOpfc_7/BiasAdd/ReadVariableOp28
fc_7/MatMul/ReadVariableOpfc_7/MatMul/ReadVariableOp2:
fc_8/BiasAdd/ReadVariableOpfc_8/BiasAdd/ReadVariableOp28
fc_8/MatMul/ReadVariableOpfc_8/MatMul/ReadVariableOp2:
fc_9/BiasAdd/ReadVariableOpfc_9/BiasAdd/ReadVariableOp28
fc_9/MatMul/ReadVariableOpfc_9/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_2_layer_call_and_return_conditional_losses_6305039

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
м	
џ
B__inference_fc_10_layer_call_and_return_conditional_losses_6305199

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
„5
э
 __inference__traced_save_6305343
file_prefix*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop*
&savev2_fc_2_kernel_read_readvariableop(
$savev2_fc_2_bias_read_readvariableop*
&savev2_fc_3_kernel_read_readvariableop(
$savev2_fc_3_bias_read_readvariableop*
&savev2_fc_4_kernel_read_readvariableop(
$savev2_fc_4_bias_read_readvariableop*
&savev2_fc_5_kernel_read_readvariableop(
$savev2_fc_5_bias_read_readvariableop*
&savev2_fc_6_kernel_read_readvariableop(
$savev2_fc_6_bias_read_readvariableop*
&savev2_fc_7_kernel_read_readvariableop(
$savev2_fc_7_bias_read_readvariableop*
&savev2_fc_8_kernel_read_readvariableop(
$savev2_fc_8_bias_read_readvariableop*
&savev2_fc_9_kernel_read_readvariableop(
$savev2_fc_9_bias_read_readvariableop+
'savev2_fc_10_kernel_read_readvariableop)
%savev2_fc_10_bias_read_readvariableop+
'savev2_fc_11_kernel_read_readvariableop)
%savev2_fc_11_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЌ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*я

value’
B“
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesВ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop&savev2_fc_3_kernel_read_readvariableop$savev2_fc_3_bias_read_readvariableop&savev2_fc_4_kernel_read_readvariableop$savev2_fc_4_bias_read_readvariableop&savev2_fc_5_kernel_read_readvariableop$savev2_fc_5_bias_read_readvariableop&savev2_fc_6_kernel_read_readvariableop$savev2_fc_6_bias_read_readvariableop&savev2_fc_7_kernel_read_readvariableop$savev2_fc_7_bias_read_readvariableop&savev2_fc_8_kernel_read_readvariableop$savev2_fc_8_bias_read_readvariableop&savev2_fc_9_kernel_read_readvariableop$savev2_fc_9_bias_read_readvariableop'savev2_fc_10_kernel_read_readvariableop%savev2_fc_10_bias_read_readvariableop'savev2_fc_11_kernel_read_readvariableop%savev2_fc_11_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*м
_input_shapesЏ
„: :
РА:А:
АА:А:
АА:А:
АА:А:
АА:А:
АА:А:	А@:@:@@:@:@ : :  : : ::
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
РА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&	"
 
_output_shapes
:
АА:!


_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:

_output_shapes
: 
зw
Ш
"__inference__wrapped_model_6304011	
input-
)mnist_fc_1_matmul_readvariableop_resource.
*mnist_fc_1_biasadd_readvariableop_resource-
)mnist_fc_2_matmul_readvariableop_resource.
*mnist_fc_2_biasadd_readvariableop_resource-
)mnist_fc_3_matmul_readvariableop_resource.
*mnist_fc_3_biasadd_readvariableop_resource-
)mnist_fc_4_matmul_readvariableop_resource.
*mnist_fc_4_biasadd_readvariableop_resource-
)mnist_fc_5_matmul_readvariableop_resource.
*mnist_fc_5_biasadd_readvariableop_resource-
)mnist_fc_6_matmul_readvariableop_resource.
*mnist_fc_6_biasadd_readvariableop_resource-
)mnist_fc_7_matmul_readvariableop_resource.
*mnist_fc_7_biasadd_readvariableop_resource-
)mnist_fc_8_matmul_readvariableop_resource.
*mnist_fc_8_biasadd_readvariableop_resource-
)mnist_fc_9_matmul_readvariableop_resource.
*mnist_fc_9_biasadd_readvariableop_resource.
*mnist_fc_10_matmul_readvariableop_resource/
+mnist_fc_10_biasadd_readvariableop_resource.
*mnist_fc_11_matmul_readvariableop_resource/
+mnist_fc_11_biasadd_readvariableop_resource/
+mnist_output_matmul_readvariableop_resource0
,mnist_output_biasadd_readvariableop_resource
identityИҐ!mnist/fc_1/BiasAdd/ReadVariableOpҐ mnist/fc_1/MatMul/ReadVariableOpҐ"mnist/fc_10/BiasAdd/ReadVariableOpҐ!mnist/fc_10/MatMul/ReadVariableOpҐ"mnist/fc_11/BiasAdd/ReadVariableOpҐ!mnist/fc_11/MatMul/ReadVariableOpҐ!mnist/fc_2/BiasAdd/ReadVariableOpҐ mnist/fc_2/MatMul/ReadVariableOpҐ!mnist/fc_3/BiasAdd/ReadVariableOpҐ mnist/fc_3/MatMul/ReadVariableOpҐ!mnist/fc_4/BiasAdd/ReadVariableOpҐ mnist/fc_4/MatMul/ReadVariableOpҐ!mnist/fc_5/BiasAdd/ReadVariableOpҐ mnist/fc_5/MatMul/ReadVariableOpҐ!mnist/fc_6/BiasAdd/ReadVariableOpҐ mnist/fc_6/MatMul/ReadVariableOpҐ!mnist/fc_7/BiasAdd/ReadVariableOpҐ mnist/fc_7/MatMul/ReadVariableOpҐ!mnist/fc_8/BiasAdd/ReadVariableOpҐ mnist/fc_8/MatMul/ReadVariableOpҐ!mnist/fc_9/BiasAdd/ReadVariableOpҐ mnist/fc_9/MatMul/ReadVariableOpҐ#mnist/output/BiasAdd/ReadVariableOpҐ"mnist/output/MatMul/ReadVariableOp{
mnist/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
mnist/flatten/ConstС
mnist/flatten/ReshapeReshapeinputmnist/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
mnist/flatten/Reshape∞
 mnist/fc_1/MatMul/ReadVariableOpReadVariableOp)mnist_fc_1_matmul_readvariableop_resource* 
_output_shapes
:
РА*
dtype02"
 mnist/fc_1/MatMul/ReadVariableOp≠
mnist/fc_1/MatMulMatMulmnist/flatten/Reshape:output:0(mnist/fc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_1/MatMulЃ
!mnist/fc_1/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!mnist/fc_1/BiasAdd/ReadVariableOpЃ
mnist/fc_1/BiasAddBiasAddmnist/fc_1/MatMul:product:0)mnist/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_1/BiasAddz
mnist/fc_1/ReluRelumnist/fc_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_1/Relu∞
 mnist/fc_2/MatMul/ReadVariableOpReadVariableOp)mnist_fc_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02"
 mnist/fc_2/MatMul/ReadVariableOpђ
mnist/fc_2/MatMulMatMulmnist/fc_1/Relu:activations:0(mnist/fc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_2/MatMulЃ
!mnist/fc_2/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!mnist/fc_2/BiasAdd/ReadVariableOpЃ
mnist/fc_2/BiasAddBiasAddmnist/fc_2/MatMul:product:0)mnist/fc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_2/BiasAddz
mnist/fc_2/ReluRelumnist/fc_2/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_2/Relu∞
 mnist/fc_3/MatMul/ReadVariableOpReadVariableOp)mnist_fc_3_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02"
 mnist/fc_3/MatMul/ReadVariableOpђ
mnist/fc_3/MatMulMatMulmnist/fc_2/Relu:activations:0(mnist/fc_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_3/MatMulЃ
!mnist/fc_3/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!mnist/fc_3/BiasAdd/ReadVariableOpЃ
mnist/fc_3/BiasAddBiasAddmnist/fc_3/MatMul:product:0)mnist/fc_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_3/BiasAddz
mnist/fc_3/ReluRelumnist/fc_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_3/Relu∞
 mnist/fc_4/MatMul/ReadVariableOpReadVariableOp)mnist_fc_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02"
 mnist/fc_4/MatMul/ReadVariableOpђ
mnist/fc_4/MatMulMatMulmnist/fc_3/Relu:activations:0(mnist/fc_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_4/MatMulЃ
!mnist/fc_4/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!mnist/fc_4/BiasAdd/ReadVariableOpЃ
mnist/fc_4/BiasAddBiasAddmnist/fc_4/MatMul:product:0)mnist/fc_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_4/BiasAddz
mnist/fc_4/ReluRelumnist/fc_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_4/Relu∞
 mnist/fc_5/MatMul/ReadVariableOpReadVariableOp)mnist_fc_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02"
 mnist/fc_5/MatMul/ReadVariableOpђ
mnist/fc_5/MatMulMatMulmnist/fc_4/Relu:activations:0(mnist/fc_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_5/MatMulЃ
!mnist/fc_5/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!mnist/fc_5/BiasAdd/ReadVariableOpЃ
mnist/fc_5/BiasAddBiasAddmnist/fc_5/MatMul:product:0)mnist/fc_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_5/BiasAddz
mnist/fc_5/ReluRelumnist/fc_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_5/Relu∞
 mnist/fc_6/MatMul/ReadVariableOpReadVariableOp)mnist_fc_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02"
 mnist/fc_6/MatMul/ReadVariableOpђ
mnist/fc_6/MatMulMatMulmnist/fc_5/Relu:activations:0(mnist/fc_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_6/MatMulЃ
!mnist/fc_6/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02#
!mnist/fc_6/BiasAdd/ReadVariableOpЃ
mnist/fc_6/BiasAddBiasAddmnist/fc_6/MatMul:product:0)mnist/fc_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_6/BiasAddz
mnist/fc_6/ReluRelumnist/fc_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
mnist/fc_6/Reluѓ
 mnist/fc_7/MatMul/ReadVariableOpReadVariableOp)mnist_fc_7_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02"
 mnist/fc_7/MatMul/ReadVariableOpЂ
mnist/fc_7/MatMulMatMulmnist/fc_6/Relu:activations:0(mnist/fc_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mnist/fc_7/MatMul≠
!mnist/fc_7/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!mnist/fc_7/BiasAdd/ReadVariableOp≠
mnist/fc_7/BiasAddBiasAddmnist/fc_7/MatMul:product:0)mnist/fc_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mnist/fc_7/BiasAddy
mnist/fc_7/ReluRelumnist/fc_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mnist/fc_7/ReluЃ
 mnist/fc_8/MatMul/ReadVariableOpReadVariableOp)mnist_fc_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02"
 mnist/fc_8/MatMul/ReadVariableOpЂ
mnist/fc_8/MatMulMatMulmnist/fc_7/Relu:activations:0(mnist/fc_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mnist/fc_8/MatMul≠
!mnist/fc_8/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!mnist/fc_8/BiasAdd/ReadVariableOp≠
mnist/fc_8/BiasAddBiasAddmnist/fc_8/MatMul:product:0)mnist/fc_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mnist/fc_8/BiasAddy
mnist/fc_8/ReluRelumnist/fc_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mnist/fc_8/ReluЃ
 mnist/fc_9/MatMul/ReadVariableOpReadVariableOp)mnist_fc_9_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02"
 mnist/fc_9/MatMul/ReadVariableOpЂ
mnist/fc_9/MatMulMatMulmnist/fc_8/Relu:activations:0(mnist/fc_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mnist/fc_9/MatMul≠
!mnist/fc_9/BiasAdd/ReadVariableOpReadVariableOp*mnist_fc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!mnist/fc_9/BiasAdd/ReadVariableOp≠
mnist/fc_9/BiasAddBiasAddmnist/fc_9/MatMul:product:0)mnist/fc_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mnist/fc_9/BiasAddy
mnist/fc_9/ReluRelumnist/fc_9/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mnist/fc_9/Relu±
!mnist/fc_10/MatMul/ReadVariableOpReadVariableOp*mnist_fc_10_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!mnist/fc_10/MatMul/ReadVariableOpЃ
mnist/fc_10/MatMulMatMulmnist/fc_9/Relu:activations:0)mnist/fc_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mnist/fc_10/MatMul∞
"mnist/fc_10/BiasAdd/ReadVariableOpReadVariableOp+mnist_fc_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"mnist/fc_10/BiasAdd/ReadVariableOp±
mnist/fc_10/BiasAddBiasAddmnist/fc_10/MatMul:product:0*mnist/fc_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mnist/fc_10/BiasAdd|
mnist/fc_10/ReluRelumnist/fc_10/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mnist/fc_10/Relu±
!mnist/fc_11/MatMul/ReadVariableOpReadVariableOp*mnist_fc_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!mnist/fc_11/MatMul/ReadVariableOpѓ
mnist/fc_11/MatMulMatMulmnist/fc_10/Relu:activations:0)mnist/fc_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
mnist/fc_11/MatMul∞
"mnist/fc_11/BiasAdd/ReadVariableOpReadVariableOp+mnist_fc_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"mnist/fc_11/BiasAdd/ReadVariableOp±
mnist/fc_11/BiasAddBiasAddmnist/fc_11/MatMul:product:0*mnist/fc_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
mnist/fc_11/BiasAdd|
mnist/fc_11/ReluRelumnist/fc_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
mnist/fc_11/Reluі
"mnist/output/MatMul/ReadVariableOpReadVariableOp+mnist_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"mnist/output/MatMul/ReadVariableOp≤
mnist/output/MatMulMatMulmnist/fc_11/Relu:activations:0*mnist/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
mnist/output/MatMul≥
#mnist/output/BiasAdd/ReadVariableOpReadVariableOp,mnist_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#mnist/output/BiasAdd/ReadVariableOpµ
mnist/output/BiasAddBiasAddmnist/output/MatMul:product:0+mnist/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
mnist/output/BiasAddИ
mnist/output/SoftmaxSoftmaxmnist/output/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
mnist/output/Softmaxќ
IdentityIdentitymnist/output/Softmax:softmax:0"^mnist/fc_1/BiasAdd/ReadVariableOp!^mnist/fc_1/MatMul/ReadVariableOp#^mnist/fc_10/BiasAdd/ReadVariableOp"^mnist/fc_10/MatMul/ReadVariableOp#^mnist/fc_11/BiasAdd/ReadVariableOp"^mnist/fc_11/MatMul/ReadVariableOp"^mnist/fc_2/BiasAdd/ReadVariableOp!^mnist/fc_2/MatMul/ReadVariableOp"^mnist/fc_3/BiasAdd/ReadVariableOp!^mnist/fc_3/MatMul/ReadVariableOp"^mnist/fc_4/BiasAdd/ReadVariableOp!^mnist/fc_4/MatMul/ReadVariableOp"^mnist/fc_5/BiasAdd/ReadVariableOp!^mnist/fc_5/MatMul/ReadVariableOp"^mnist/fc_6/BiasAdd/ReadVariableOp!^mnist/fc_6/MatMul/ReadVariableOp"^mnist/fc_7/BiasAdd/ReadVariableOp!^mnist/fc_7/MatMul/ReadVariableOp"^mnist/fc_8/BiasAdd/ReadVariableOp!^mnist/fc_8/MatMul/ReadVariableOp"^mnist/fc_9/BiasAdd/ReadVariableOp!^mnist/fc_9/MatMul/ReadVariableOp$^mnist/output/BiasAdd/ReadVariableOp#^mnist/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2F
!mnist/fc_1/BiasAdd/ReadVariableOp!mnist/fc_1/BiasAdd/ReadVariableOp2D
 mnist/fc_1/MatMul/ReadVariableOp mnist/fc_1/MatMul/ReadVariableOp2H
"mnist/fc_10/BiasAdd/ReadVariableOp"mnist/fc_10/BiasAdd/ReadVariableOp2F
!mnist/fc_10/MatMul/ReadVariableOp!mnist/fc_10/MatMul/ReadVariableOp2H
"mnist/fc_11/BiasAdd/ReadVariableOp"mnist/fc_11/BiasAdd/ReadVariableOp2F
!mnist/fc_11/MatMul/ReadVariableOp!mnist/fc_11/MatMul/ReadVariableOp2F
!mnist/fc_2/BiasAdd/ReadVariableOp!mnist/fc_2/BiasAdd/ReadVariableOp2D
 mnist/fc_2/MatMul/ReadVariableOp mnist/fc_2/MatMul/ReadVariableOp2F
!mnist/fc_3/BiasAdd/ReadVariableOp!mnist/fc_3/BiasAdd/ReadVariableOp2D
 mnist/fc_3/MatMul/ReadVariableOp mnist/fc_3/MatMul/ReadVariableOp2F
!mnist/fc_4/BiasAdd/ReadVariableOp!mnist/fc_4/BiasAdd/ReadVariableOp2D
 mnist/fc_4/MatMul/ReadVariableOp mnist/fc_4/MatMul/ReadVariableOp2F
!mnist/fc_5/BiasAdd/ReadVariableOp!mnist/fc_5/BiasAdd/ReadVariableOp2D
 mnist/fc_5/MatMul/ReadVariableOp mnist/fc_5/MatMul/ReadVariableOp2F
!mnist/fc_6/BiasAdd/ReadVariableOp!mnist/fc_6/BiasAdd/ReadVariableOp2D
 mnist/fc_6/MatMul/ReadVariableOp mnist/fc_6/MatMul/ReadVariableOp2F
!mnist/fc_7/BiasAdd/ReadVariableOp!mnist/fc_7/BiasAdd/ReadVariableOp2D
 mnist/fc_7/MatMul/ReadVariableOp mnist/fc_7/MatMul/ReadVariableOp2F
!mnist/fc_8/BiasAdd/ReadVariableOp!mnist/fc_8/BiasAdd/ReadVariableOp2D
 mnist/fc_8/MatMul/ReadVariableOp mnist/fc_8/MatMul/ReadVariableOp2F
!mnist/fc_9/BiasAdd/ReadVariableOp!mnist/fc_9/BiasAdd/ReadVariableOp2D
 mnist/fc_9/MatMul/ReadVariableOp mnist/fc_9/MatMul/ReadVariableOp2J
#mnist/output/BiasAdd/ReadVariableOp#mnist/output/BiasAdd/ReadVariableOp2H
"mnist/output/MatMul/ReadVariableOp"mnist/output/MatMul/ReadVariableOp:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameinput
І
E
)__inference_flatten_layer_call_fn_6305008

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_63040212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
{
&__inference_fc_6_layer_call_fn_6305128

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_63041752
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
л	
Џ
A__inference_fc_8_layer_call_and_return_conditional_losses_6304229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
«
”
'__inference_mnist_layer_call_fn_6304944

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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_63044872
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_1_layer_call_and_return_conditional_losses_6305019

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
РА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
Ќh
Щ
B__inference_mnist_layer_call_and_return_conditional_losses_6304891

inputs'
#fc_1_matmul_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource'
#fc_2_matmul_readvariableop_resource(
$fc_2_biasadd_readvariableop_resource'
#fc_3_matmul_readvariableop_resource(
$fc_3_biasadd_readvariableop_resource'
#fc_4_matmul_readvariableop_resource(
$fc_4_biasadd_readvariableop_resource'
#fc_5_matmul_readvariableop_resource(
$fc_5_biasadd_readvariableop_resource'
#fc_6_matmul_readvariableop_resource(
$fc_6_biasadd_readvariableop_resource'
#fc_7_matmul_readvariableop_resource(
$fc_7_biasadd_readvariableop_resource'
#fc_8_matmul_readvariableop_resource(
$fc_8_biasadd_readvariableop_resource'
#fc_9_matmul_readvariableop_resource(
$fc_9_biasadd_readvariableop_resource(
$fc_10_matmul_readvariableop_resource)
%fc_10_biasadd_readvariableop_resource(
$fc_11_matmul_readvariableop_resource)
%fc_11_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityИҐfc_1/BiasAdd/ReadVariableOpҐfc_1/MatMul/ReadVariableOpҐfc_10/BiasAdd/ReadVariableOpҐfc_10/MatMul/ReadVariableOpҐfc_11/BiasAdd/ReadVariableOpҐfc_11/MatMul/ReadVariableOpҐfc_2/BiasAdd/ReadVariableOpҐfc_2/MatMul/ReadVariableOpҐfc_3/BiasAdd/ReadVariableOpҐfc_3/MatMul/ReadVariableOpҐfc_4/BiasAdd/ReadVariableOpҐfc_4/MatMul/ReadVariableOpҐfc_5/BiasAdd/ReadVariableOpҐfc_5/MatMul/ReadVariableOpҐfc_6/BiasAdd/ReadVariableOpҐfc_6/MatMul/ReadVariableOpҐfc_7/BiasAdd/ReadVariableOpҐfc_7/MatMul/ReadVariableOpҐfc_8/BiasAdd/ReadVariableOpҐfc_8/MatMul/ReadVariableOpҐfc_9/BiasAdd/ReadVariableOpҐfc_9/MatMul/ReadVariableOpҐoutput/BiasAdd/ReadVariableOpҐoutput/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
flatten/ConstА
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
flatten/ReshapeЮ
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource* 
_output_shapes
:
РА*
dtype02
fc_1/MatMul/ReadVariableOpХ
fc_1/MatMulMatMulflatten/Reshape:output:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_1/MatMulЬ
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_1/BiasAdd/ReadVariableOpЦ
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_1/BiasAddh
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_1/ReluЮ
fc_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_2/MatMul/ReadVariableOpФ
fc_2/MatMulMatMulfc_1/Relu:activations:0"fc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_2/MatMulЬ
fc_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_2/BiasAdd/ReadVariableOpЦ
fc_2/BiasAddBiasAddfc_2/MatMul:product:0#fc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_2/BiasAddh
	fc_2/ReluRelufc_2/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_2/ReluЮ
fc_3/MatMul/ReadVariableOpReadVariableOp#fc_3_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_3/MatMul/ReadVariableOpФ
fc_3/MatMulMatMulfc_2/Relu:activations:0"fc_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_3/MatMulЬ
fc_3/BiasAdd/ReadVariableOpReadVariableOp$fc_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_3/BiasAdd/ReadVariableOpЦ
fc_3/BiasAddBiasAddfc_3/MatMul:product:0#fc_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_3/BiasAddh
	fc_3/ReluRelufc_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_3/ReluЮ
fc_4/MatMul/ReadVariableOpReadVariableOp#fc_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_4/MatMul/ReadVariableOpФ
fc_4/MatMulMatMulfc_3/Relu:activations:0"fc_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_4/MatMulЬ
fc_4/BiasAdd/ReadVariableOpReadVariableOp$fc_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_4/BiasAdd/ReadVariableOpЦ
fc_4/BiasAddBiasAddfc_4/MatMul:product:0#fc_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_4/BiasAddh
	fc_4/ReluRelufc_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_4/ReluЮ
fc_5/MatMul/ReadVariableOpReadVariableOp#fc_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_5/MatMul/ReadVariableOpФ
fc_5/MatMulMatMulfc_4/Relu:activations:0"fc_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_5/MatMulЬ
fc_5/BiasAdd/ReadVariableOpReadVariableOp$fc_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_5/BiasAdd/ReadVariableOpЦ
fc_5/BiasAddBiasAddfc_5/MatMul:product:0#fc_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_5/BiasAddh
	fc_5/ReluRelufc_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_5/ReluЮ
fc_6/MatMul/ReadVariableOpReadVariableOp#fc_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
fc_6/MatMul/ReadVariableOpФ
fc_6/MatMulMatMulfc_5/Relu:activations:0"fc_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_6/MatMulЬ
fc_6/BiasAdd/ReadVariableOpReadVariableOp$fc_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
fc_6/BiasAdd/ReadVariableOpЦ
fc_6/BiasAddBiasAddfc_6/MatMul:product:0#fc_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
fc_6/BiasAddh
	fc_6/ReluRelufc_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
	fc_6/ReluЭ
fc_7/MatMul/ReadVariableOpReadVariableOp#fc_7_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
fc_7/MatMul/ReadVariableOpУ
fc_7/MatMulMatMulfc_6/Relu:activations:0"fc_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_7/MatMulЫ
fc_7/BiasAdd/ReadVariableOpReadVariableOp$fc_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_7/BiasAdd/ReadVariableOpХ
fc_7/BiasAddBiasAddfc_7/MatMul:product:0#fc_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_7/BiasAddg
	fc_7/ReluRelufc_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	fc_7/ReluЬ
fc_8/MatMul/ReadVariableOpReadVariableOp#fc_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
fc_8/MatMul/ReadVariableOpУ
fc_8/MatMulMatMulfc_7/Relu:activations:0"fc_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_8/MatMulЫ
fc_8/BiasAdd/ReadVariableOpReadVariableOp$fc_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc_8/BiasAdd/ReadVariableOpХ
fc_8/BiasAddBiasAddfc_8/MatMul:product:0#fc_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
fc_8/BiasAddg
	fc_8/ReluRelufc_8/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	fc_8/ReluЬ
fc_9/MatMul/ReadVariableOpReadVariableOp#fc_9_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
fc_9/MatMul/ReadVariableOpУ
fc_9/MatMulMatMulfc_8/Relu:activations:0"fc_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_9/MatMulЫ
fc_9/BiasAdd/ReadVariableOpReadVariableOp$fc_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_9/BiasAdd/ReadVariableOpХ
fc_9/BiasAddBiasAddfc_9/MatMul:product:0#fc_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_9/BiasAddg
	fc_9/ReluRelufc_9/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
	fc_9/ReluЯ
fc_10/MatMul/ReadVariableOpReadVariableOp$fc_10_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
fc_10/MatMul/ReadVariableOpЦ
fc_10/MatMulMatMulfc_9/Relu:activations:0#fc_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_10/MatMulЮ
fc_10/BiasAdd/ReadVariableOpReadVariableOp%fc_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
fc_10/BiasAdd/ReadVariableOpЩ
fc_10/BiasAddBiasAddfc_10/MatMul:product:0$fc_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
fc_10/BiasAddj

fc_10/ReluRelufc_10/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

fc_10/ReluЯ
fc_11/MatMul/ReadVariableOpReadVariableOp$fc_11_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
fc_11/MatMul/ReadVariableOpЧ
fc_11/MatMulMatMulfc_10/Relu:activations:0#fc_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
fc_11/MatMulЮ
fc_11/BiasAdd/ReadVariableOpReadVariableOp%fc_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_11/BiasAdd/ReadVariableOpЩ
fc_11/BiasAddBiasAddfc_11/MatMul:product:0$fc_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
fc_11/BiasAddj

fc_11/ReluRelufc_11/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

fc_11/ReluҐ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
output/MatMul/ReadVariableOpЪ
output/MatMulMatMulfc_11/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
output/MatMul°
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
output/SoftmaxЄ
IdentityIdentityoutput/Softmax:softmax:0^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^fc_10/BiasAdd/ReadVariableOp^fc_10/MatMul/ReadVariableOp^fc_11/BiasAdd/ReadVariableOp^fc_11/MatMul/ReadVariableOp^fc_2/BiasAdd/ReadVariableOp^fc_2/MatMul/ReadVariableOp^fc_3/BiasAdd/ReadVariableOp^fc_3/MatMul/ReadVariableOp^fc_4/BiasAdd/ReadVariableOp^fc_4/MatMul/ReadVariableOp^fc_5/BiasAdd/ReadVariableOp^fc_5/MatMul/ReadVariableOp^fc_6/BiasAdd/ReadVariableOp^fc_6/MatMul/ReadVariableOp^fc_7/BiasAdd/ReadVariableOp^fc_7/MatMul/ReadVariableOp^fc_8/BiasAdd/ReadVariableOp^fc_8/MatMul/ReadVariableOp^fc_9/BiasAdd/ReadVariableOp^fc_9/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2<
fc_10/BiasAdd/ReadVariableOpfc_10/BiasAdd/ReadVariableOp2:
fc_10/MatMul/ReadVariableOpfc_10/MatMul/ReadVariableOp2<
fc_11/BiasAdd/ReadVariableOpfc_11/BiasAdd/ReadVariableOp2:
fc_11/MatMul/ReadVariableOpfc_11/MatMul/ReadVariableOp2:
fc_2/BiasAdd/ReadVariableOpfc_2/BiasAdd/ReadVariableOp28
fc_2/MatMul/ReadVariableOpfc_2/MatMul/ReadVariableOp2:
fc_3/BiasAdd/ReadVariableOpfc_3/BiasAdd/ReadVariableOp28
fc_3/MatMul/ReadVariableOpfc_3/MatMul/ReadVariableOp2:
fc_4/BiasAdd/ReadVariableOpfc_4/BiasAdd/ReadVariableOp28
fc_4/MatMul/ReadVariableOpfc_4/MatMul/ReadVariableOp2:
fc_5/BiasAdd/ReadVariableOpfc_5/BiasAdd/ReadVariableOp28
fc_5/MatMul/ReadVariableOpfc_5/MatMul/ReadVariableOp2:
fc_6/BiasAdd/ReadVariableOpfc_6/BiasAdd/ReadVariableOp28
fc_6/MatMul/ReadVariableOpfc_6/MatMul/ReadVariableOp2:
fc_7/BiasAdd/ReadVariableOpfc_7/BiasAdd/ReadVariableOp28
fc_7/MatMul/ReadVariableOpfc_7/MatMul/ReadVariableOp2:
fc_8/BiasAdd/ReadVariableOpfc_8/BiasAdd/ReadVariableOp28
fc_8/MatMul/ReadVariableOpfc_8/MatMul/ReadVariableOp2:
fc_9/BiasAdd/ReadVariableOpfc_9/BiasAdd/ReadVariableOp28
fc_9/MatMul/ReadVariableOpfc_9/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
л	
Џ
A__inference_fc_8_layer_call_and_return_conditional_losses_6305159

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
”=
С
B__inference_mnist_layer_call_and_return_conditional_losses_6304487

inputs
fc_1_6304426
fc_1_6304428
fc_2_6304431
fc_2_6304433
fc_3_6304436
fc_3_6304438
fc_4_6304441
fc_4_6304443
fc_5_6304446
fc_5_6304448
fc_6_6304451
fc_6_6304453
fc_7_6304456
fc_7_6304458
fc_8_6304461
fc_8_6304463
fc_9_6304466
fc_9_6304468
fc_10_6304471
fc_10_6304473
fc_11_6304476
fc_11_6304478
output_6304481
output_6304483
identityИҐfc_1/StatefulPartitionedCallҐfc_10/StatefulPartitionedCallҐfc_11/StatefulPartitionedCallҐfc_2/StatefulPartitionedCallҐfc_3/StatefulPartitionedCallҐfc_4/StatefulPartitionedCallҐfc_5/StatefulPartitionedCallҐfc_6/StatefulPartitionedCallҐfc_7/StatefulPartitionedCallҐfc_8/StatefulPartitionedCallҐfc_9/StatefulPartitionedCallҐoutput/StatefulPartitionedCall÷
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_63040212
flatten/PartitionedCall°
fc_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1_6304426fc_1_6304428*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_63040402
fc_1/StatefulPartitionedCall¶
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_6304431fc_2_6304433*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_63040672
fc_2/StatefulPartitionedCall¶
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0fc_3_6304436fc_3_6304438*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_63040942
fc_3/StatefulPartitionedCall¶
fc_4/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0fc_4_6304441fc_4_6304443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_63041212
fc_4/StatefulPartitionedCall¶
fc_5/StatefulPartitionedCallStatefulPartitionedCall%fc_4/StatefulPartitionedCall:output:0fc_5_6304446fc_5_6304448*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_63041482
fc_5/StatefulPartitionedCall¶
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_6304451fc_6_6304453*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_63041752
fc_6/StatefulPartitionedCall•
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0fc_7_6304456fc_7_6304458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_63042022
fc_7/StatefulPartitionedCall•
fc_8/StatefulPartitionedCallStatefulPartitionedCall%fc_7/StatefulPartitionedCall:output:0fc_8_6304461fc_8_6304463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_63042292
fc_8/StatefulPartitionedCall•
fc_9/StatefulPartitionedCallStatefulPartitionedCall%fc_8/StatefulPartitionedCall:output:0fc_9_6304466fc_9_6304468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_63042562
fc_9/StatefulPartitionedCall™
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_6304471fc_10_6304473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_63042832
fc_10/StatefulPartitionedCallЂ
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0fc_11_6304476fc_11_6304478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_63043102
fc_11/StatefulPartitionedCall∞
output/StatefulPartitionedCallStatefulPartitionedCall&fc_11/StatefulPartitionedCall:output:0output_6304481output_6304483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_63043372 
output/StatefulPartitionedCallу
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall^fc_10/StatefulPartitionedCall^fc_11/StatefulPartitionedCall^fc_2/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^fc_4/StatefulPartitionedCall^fc_5/StatefulPartitionedCall^fc_6/StatefulPartitionedCall^fc_7/StatefulPartitionedCall^fc_8/StatefulPartitionedCall^fc_9/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2>
fc_11/StatefulPartitionedCallfc_11/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_4/StatefulPartitionedCallfc_4/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2<
fc_8/StatefulPartitionedCallfc_8/StatefulPartitionedCall2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
–=
Р
B__inference_mnist_layer_call_and_return_conditional_losses_6304354	
input
fc_1_6304051
fc_1_6304053
fc_2_6304078
fc_2_6304080
fc_3_6304105
fc_3_6304107
fc_4_6304132
fc_4_6304134
fc_5_6304159
fc_5_6304161
fc_6_6304186
fc_6_6304188
fc_7_6304213
fc_7_6304215
fc_8_6304240
fc_8_6304242
fc_9_6304267
fc_9_6304269
fc_10_6304294
fc_10_6304296
fc_11_6304321
fc_11_6304323
output_6304348
output_6304350
identityИҐfc_1/StatefulPartitionedCallҐfc_10/StatefulPartitionedCallҐfc_11/StatefulPartitionedCallҐfc_2/StatefulPartitionedCallҐfc_3/StatefulPartitionedCallҐfc_4/StatefulPartitionedCallҐfc_5/StatefulPartitionedCallҐfc_6/StatefulPartitionedCallҐfc_7/StatefulPartitionedCallҐfc_8/StatefulPartitionedCallҐfc_9/StatefulPartitionedCallҐoutput/StatefulPartitionedCall’
flatten/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_63040212
flatten/PartitionedCall°
fc_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1_6304051fc_1_6304053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_63040402
fc_1/StatefulPartitionedCall¶
fc_2/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0fc_2_6304078fc_2_6304080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_2_layer_call_and_return_conditional_losses_63040672
fc_2/StatefulPartitionedCall¶
fc_3/StatefulPartitionedCallStatefulPartitionedCall%fc_2/StatefulPartitionedCall:output:0fc_3_6304105fc_3_6304107*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_3_layer_call_and_return_conditional_losses_63040942
fc_3/StatefulPartitionedCall¶
fc_4/StatefulPartitionedCallStatefulPartitionedCall%fc_3/StatefulPartitionedCall:output:0fc_4_6304132fc_4_6304134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_4_layer_call_and_return_conditional_losses_63041212
fc_4/StatefulPartitionedCall¶
fc_5/StatefulPartitionedCallStatefulPartitionedCall%fc_4/StatefulPartitionedCall:output:0fc_5_6304159fc_5_6304161*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_63041482
fc_5/StatefulPartitionedCall¶
fc_6/StatefulPartitionedCallStatefulPartitionedCall%fc_5/StatefulPartitionedCall:output:0fc_6_6304186fc_6_6304188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_6_layer_call_and_return_conditional_losses_63041752
fc_6/StatefulPartitionedCall•
fc_7/StatefulPartitionedCallStatefulPartitionedCall%fc_6/StatefulPartitionedCall:output:0fc_7_6304213fc_7_6304215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_7_layer_call_and_return_conditional_losses_63042022
fc_7/StatefulPartitionedCall•
fc_8/StatefulPartitionedCallStatefulPartitionedCall%fc_7/StatefulPartitionedCall:output:0fc_8_6304240fc_8_6304242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_8_layer_call_and_return_conditional_losses_63042292
fc_8/StatefulPartitionedCall•
fc_9/StatefulPartitionedCallStatefulPartitionedCall%fc_8/StatefulPartitionedCall:output:0fc_9_6304267fc_9_6304269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_63042562
fc_9/StatefulPartitionedCall™
fc_10/StatefulPartitionedCallStatefulPartitionedCall%fc_9/StatefulPartitionedCall:output:0fc_10_6304294fc_10_6304296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_10_layer_call_and_return_conditional_losses_63042832
fc_10/StatefulPartitionedCallЂ
fc_11/StatefulPartitionedCallStatefulPartitionedCall&fc_10/StatefulPartitionedCall:output:0fc_11_6304321fc_11_6304323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_fc_11_layer_call_and_return_conditional_losses_63043102
fc_11/StatefulPartitionedCall∞
output/StatefulPartitionedCallStatefulPartitionedCall&fc_11/StatefulPartitionedCall:output:0output_6304348output_6304350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_63043372 
output/StatefulPartitionedCallу
IdentityIdentity'output/StatefulPartitionedCall:output:0^fc_1/StatefulPartitionedCall^fc_10/StatefulPartitionedCall^fc_11/StatefulPartitionedCall^fc_2/StatefulPartitionedCall^fc_3/StatefulPartitionedCall^fc_4/StatefulPartitionedCall^fc_5/StatefulPartitionedCall^fc_6/StatefulPartitionedCall^fc_7/StatefulPartitionedCall^fc_8/StatefulPartitionedCall^fc_9/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2>
fc_10/StatefulPartitionedCallfc_10/StatefulPartitionedCall2>
fc_11/StatefulPartitionedCallfc_11/StatefulPartitionedCall2<
fc_2/StatefulPartitionedCallfc_2/StatefulPartitionedCall2<
fc_3/StatefulPartitionedCallfc_3/StatefulPartitionedCall2<
fc_4/StatefulPartitionedCallfc_4/StatefulPartitionedCall2<
fc_5/StatefulPartitionedCallfc_5/StatefulPartitionedCall2<
fc_6/StatefulPartitionedCallfc_6/StatefulPartitionedCall2<
fc_7/StatefulPartitionedCallfc_7/StatefulPartitionedCall2<
fc_8/StatefulPartitionedCallfc_8/StatefulPartitionedCall2<
fc_9/StatefulPartitionedCallfc_9/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameinput
шb
™
#__inference__traced_restore_6305425
file_prefix 
assignvariableop_fc_1_kernel 
assignvariableop_1_fc_1_bias"
assignvariableop_2_fc_2_kernel 
assignvariableop_3_fc_2_bias"
assignvariableop_4_fc_3_kernel 
assignvariableop_5_fc_3_bias"
assignvariableop_6_fc_4_kernel 
assignvariableop_7_fc_4_bias"
assignvariableop_8_fc_5_kernel 
assignvariableop_9_fc_5_bias#
assignvariableop_10_fc_6_kernel!
assignvariableop_11_fc_6_bias#
assignvariableop_12_fc_7_kernel!
assignvariableop_13_fc_7_bias#
assignvariableop_14_fc_8_kernel!
assignvariableop_15_fc_8_bias#
assignvariableop_16_fc_9_kernel!
assignvariableop_17_fc_9_bias$
 assignvariableop_18_fc_10_kernel"
assignvariableop_19_fc_10_bias$
 assignvariableop_20_fc_11_kernel"
assignvariableop_21_fc_11_bias%
!assignvariableop_22_output_kernel#
assignvariableop_23_output_bias
identity_25ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9”
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*я

value’
B“
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesј
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices®
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЫ
AssignVariableOpAssignVariableOpassignvariableop_fc_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1°
AssignVariableOp_1AssignVariableOpassignvariableop_1_fc_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_fc_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3°
AssignVariableOp_3AssignVariableOpassignvariableop_3_fc_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_fc_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOpassignvariableop_5_fc_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7°
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_fc_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9°
AssignVariableOp_9AssignVariableOpassignvariableop_9_fc_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10І
AssignVariableOp_10AssignVariableOpassignvariableop_10_fc_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11•
AssignVariableOp_11AssignVariableOpassignvariableop_11_fc_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12І
AssignVariableOp_12AssignVariableOpassignvariableop_12_fc_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13•
AssignVariableOp_13AssignVariableOpassignvariableop_13_fc_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14І
AssignVariableOp_14AssignVariableOpassignvariableop_14_fc_8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15•
AssignVariableOp_15AssignVariableOpassignvariableop_15_fc_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16І
AssignVariableOp_16AssignVariableOpassignvariableop_16_fc_9_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17•
AssignVariableOp_17AssignVariableOpassignvariableop_17_fc_9_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp assignvariableop_18_fc_10_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¶
AssignVariableOp_19AssignVariableOpassignvariableop_19_fc_10_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp assignvariableop_20_fc_11_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¶
AssignVariableOp_21AssignVariableOpassignvariableop_21_fc_11_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22©
AssignVariableOp_22AssignVariableOp!assignvariableop_22_output_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23І
AssignVariableOp_23AssignVariableOpassignvariableop_23_output_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpо
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24б
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
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
ф	
Џ
A__inference_fc_4_layer_call_and_return_conditional_losses_6305079

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
х	
№
C__inference_output_layer_call_and_return_conditional_losses_6305239

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
{
&__inference_fc_5_layer_call_fn_6305108

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_5_layer_call_and_return_conditional_losses_63041482
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_5_layer_call_and_return_conditional_losses_6305099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ё
}
(__inference_output_layer_call_fn_6305248

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_63043372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м	
џ
B__inference_fc_11_layer_call_and_return_conditional_losses_6305219

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ф	
Џ
A__inference_fc_1_layer_call_and_return_conditional_losses_6304040

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
РА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ƒ
“
'__inference_mnist_layer_call_fn_6304538	
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

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_mnist_layer_call_and_return_conditional_losses_63044872
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*О
_input_shapes}
{:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_nameinput
ф	
Џ
A__inference_fc_3_layer_call_and_return_conditional_losses_6305059

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ў
{
&__inference_fc_9_layer_call_fn_6305188

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_fc_9_layer_call_and_return_conditional_losses_63042562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≠
serving_defaultЩ
?
input6
serving_default_input:0€€€€€€€€€:
output0
StatefulPartitionedCall:0€€€€€€€€€
tensorflow/serving/predict:†В
≥l
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+¶&call_and_return_all_conditional_losses
І_default_save_signature
®__call__"Аg
_tf_keras_networkдf{"class_name": "Functional", "name": "mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_2", "inbound_nodes": [[["fc_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_3", "inbound_nodes": [[["fc_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_4", "inbound_nodes": [[["fc_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_5", "inbound_nodes": [[["fc_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_6", "inbound_nodes": [[["fc_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_7", "inbound_nodes": [[["fc_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_8", "inbound_nodes": [[["fc_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_9", "inbound_nodes": [[["fc_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_10", "inbound_nodes": [[["fc_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_11", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_11", "inbound_nodes": [[["fc_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["fc_11", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_2", "inbound_nodes": [[["fc_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_3", "inbound_nodes": [[["fc_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_4", "inbound_nodes": [[["fc_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_5", "inbound_nodes": [[["fc_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_6", "inbound_nodes": [[["fc_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_7", "inbound_nodes": [[["fc_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_8", "inbound_nodes": [[["fc_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_9", "inbound_nodes": [[["fc_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_10", "inbound_nodes": [[["fc_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_11", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_11", "inbound_nodes": [[["fc_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["fc_11", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}}
х"т
_tf_keras_input_layer“{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
д
	variables
trainable_variables
regularization_losses
	keras_api
+©&call_and_return_all_conditional_losses
™__call__"”
_tf_keras_layerє{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
п

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+Ђ&call_and_return_all_conditional_losses
ђ__call__"»
_tf_keras_layerЃ{"class_name": "Dense", "name": "fc_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
п

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
+≠&call_and_return_all_conditional_losses
Ѓ__call__"»
_tf_keras_layerЃ{"class_name": "Dense", "name": "fc_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
п

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+ѓ&call_and_return_all_conditional_losses
∞__call__"»
_tf_keras_layerЃ{"class_name": "Dense", "name": "fc_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
п

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+±&call_and_return_all_conditional_losses
≤__call__"»
_tf_keras_layerЃ{"class_name": "Dense", "name": "fc_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
п

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
+≥&call_and_return_all_conditional_losses
і__call__"»
_tf_keras_layerЃ{"class_name": "Dense", "name": "fc_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
п

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
+µ&call_and_return_all_conditional_losses
ґ__call__"»
_tf_keras_layerЃ{"class_name": "Dense", "name": "fc_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
о

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
+Ј&call_and_return_all_conditional_losses
Є__call__"«
_tf_keras_layer≠{"class_name": "Dense", "name": "fc_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
м

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"≈
_tf_keras_layerЂ{"class_name": "Dense", "name": "fc_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_8", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
м

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+ї&call_and_return_all_conditional_losses
Љ__call__"≈
_tf_keras_layerЂ{"class_name": "Dense", "name": "fc_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_9", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
о

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+љ&call_and_return_all_conditional_losses
Њ__call__"«
_tf_keras_layer≠{"class_name": "Dense", "name": "fc_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
о

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
+њ&call_and_return_all_conditional_losses
ј__call__"«
_tf_keras_layer≠{"class_name": "Dense", "name": "fc_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fc_11", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
у

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+Ѕ&call_and_return_all_conditional_losses
¬__call__"ћ
_tf_keras_layer≤{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
÷
0
1
2
3
$4
%5
*6
+7
08
19
610
711
<12
=13
B14
C15
H16
I17
N18
O19
T20
U21
Z22
[23"
trackable_list_wrapper
÷
0
1
2
3
$4
%5
*6
+7
08
19
610
711
<12
=13
B14
C15
H16
I17
N18
O19
T20
U21
Z22
[23"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
`layer_metrics
	variables
anon_trainable_variables
bmetrics

clayers
dlayer_regularization_losses
trainable_variables
regularization_losses
®__call__
І_default_save_signature
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
-
√serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
elayer_metrics
	variables
fmetrics
gnon_trainable_variables

hlayers
ilayer_regularization_losses
trainable_variables
regularization_losses
™__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
:
РА2fc_1/kernel
:А2	fc_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
jlayer_metrics
	variables
kmetrics
lnon_trainable_variables

mlayers
nlayer_regularization_losses
trainable_variables
regularization_losses
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
:
АА2fc_2/kernel
:А2	fc_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
olayer_metrics
 	variables
pmetrics
qnon_trainable_variables

rlayers
slayer_regularization_losses
!trainable_variables
"regularization_losses
Ѓ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
:
АА2fc_3/kernel
:А2	fc_3/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
tlayer_metrics
&	variables
umetrics
vnon_trainable_variables

wlayers
xlayer_regularization_losses
'trainable_variables
(regularization_losses
∞__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
:
АА2fc_4/kernel
:А2	fc_4/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
ylayer_metrics
,	variables
zmetrics
{non_trainable_variables

|layers
}layer_regularization_losses
-trainable_variables
.regularization_losses
≤__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
:
АА2fc_5/kernel
:А2	fc_5/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
≥
~layer_metrics
2	variables
metrics
Аnon_trainable_variables
Бlayers
 Вlayer_regularization_losses
3trainable_variables
4regularization_losses
і__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
:
АА2fc_6/kernel
:А2	fc_6/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Гlayer_metrics
8	variables
Дmetrics
Еnon_trainable_variables
Жlayers
 Зlayer_regularization_losses
9trainable_variables
:regularization_losses
ґ__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
:	А@2fc_7/kernel
:@2	fc_7/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Иlayer_metrics
>	variables
Йmetrics
Кnon_trainable_variables
Лlayers
 Мlayer_regularization_losses
?trainable_variables
@regularization_losses
Є__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
:@@2fc_8/kernel
:@2	fc_8/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Нlayer_metrics
D	variables
Оmetrics
Пnon_trainable_variables
Рlayers
 Сlayer_regularization_losses
Etrainable_variables
Fregularization_losses
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
:@ 2fc_9/kernel
: 2	fc_9/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Тlayer_metrics
J	variables
Уmetrics
Фnon_trainable_variables
Хlayers
 Цlayer_regularization_losses
Ktrainable_variables
Lregularization_losses
Љ__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
:  2fc_10/kernel
: 2
fc_10/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Чlayer_metrics
P	variables
Шmetrics
Щnon_trainable_variables
Ъlayers
 Ыlayer_regularization_losses
Qtrainable_variables
Rregularization_losses
Њ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
: 2fc_11/kernel
:2
fc_11/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ьlayer_metrics
V	variables
Эmetrics
Юnon_trainable_variables
Яlayers
 †layer_regularization_losses
Wtrainable_variables
Xregularization_losses
ј__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
:
2output/kernel
:
2output/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
°layer_metrics
\	variables
Ґmetrics
£non_trainable_variables
§layers
 •layer_regularization_losses
]trainable_variables
^regularization_losses
¬__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ж
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
13"
trackable_list_wrapper
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
trackable_list_wrapper
÷2”
B__inference_mnist_layer_call_and_return_conditional_losses_6304891
B__inference_mnist_layer_call_and_return_conditional_losses_6304801
B__inference_mnist_layer_call_and_return_conditional_losses_6304419
B__inference_mnist_layer_call_and_return_conditional_losses_6304354ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
"__inference__wrapped_model_6304011Љ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *,Ґ)
'К$
input€€€€€€€€€
к2з
'__inference_mnist_layer_call_fn_6304997
'__inference_mnist_layer_call_fn_6304944
'__inference_mnist_layer_call_fn_6304656
'__inference_mnist_layer_call_fn_6304538ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_flatten_layer_call_and_return_conditional_losses_6305003Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_flatten_layer_call_fn_6305008Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_1_layer_call_and_return_conditional_losses_6305019Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_1_layer_call_fn_6305028Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_2_layer_call_and_return_conditional_losses_6305039Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_2_layer_call_fn_6305048Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_3_layer_call_and_return_conditional_losses_6305059Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_3_layer_call_fn_6305068Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_4_layer_call_and_return_conditional_losses_6305079Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_4_layer_call_fn_6305088Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_5_layer_call_and_return_conditional_losses_6305099Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_5_layer_call_fn_6305108Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_6_layer_call_and_return_conditional_losses_6305119Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_6_layer_call_fn_6305128Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_7_layer_call_and_return_conditional_losses_6305139Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_7_layer_call_fn_6305148Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_8_layer_call_and_return_conditional_losses_6305159Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_8_layer_call_fn_6305168Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_fc_9_layer_call_and_return_conditional_losses_6305179Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_fc_9_layer_call_fn_6305188Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_fc_10_layer_call_and_return_conditional_losses_6305199Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_fc_10_layer_call_fn_6305208Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_fc_11_layer_call_and_return_conditional_losses_6305219Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_fc_11_layer_call_fn_6305228Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_output_layer_call_and_return_conditional_losses_6305239Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_output_layer_call_fn_6305248Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 B«
%__inference_signature_wrapper_6304711input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ™
"__inference__wrapped_model_6304011Г$%*+0167<=BCHINOTUZ[6Ґ3
,Ґ)
'К$
input€€€€€€€€€
™ "/™,
*
output К
output€€€€€€€€€
Ґ
B__inference_fc_10_layer_call_and_return_conditional_losses_6305199\NO/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ z
'__inference_fc_10_layer_call_fn_6305208ONO/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ Ґ
B__inference_fc_11_layer_call_and_return_conditional_losses_6305219\TU/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
'__inference_fc_11_layer_call_fn_6305228OTU/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€£
A__inference_fc_1_layer_call_and_return_conditional_losses_6305019^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_fc_1_layer_call_fn_6305028Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "К€€€€€€€€€А£
A__inference_fc_2_layer_call_and_return_conditional_losses_6305039^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_fc_2_layer_call_fn_6305048Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А£
A__inference_fc_3_layer_call_and_return_conditional_losses_6305059^$%0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_fc_3_layer_call_fn_6305068Q$%0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А£
A__inference_fc_4_layer_call_and_return_conditional_losses_6305079^*+0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_fc_4_layer_call_fn_6305088Q*+0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А£
A__inference_fc_5_layer_call_and_return_conditional_losses_6305099^010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_fc_5_layer_call_fn_6305108Q010Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А£
A__inference_fc_6_layer_call_and_return_conditional_losses_6305119^670Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ {
&__inference_fc_6_layer_call_fn_6305128Q670Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€АҐ
A__inference_fc_7_layer_call_and_return_conditional_losses_6305139]<=0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ z
&__inference_fc_7_layer_call_fn_6305148P<=0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@°
A__inference_fc_8_layer_call_and_return_conditional_losses_6305159\BC/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ y
&__inference_fc_8_layer_call_fn_6305168OBC/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@°
A__inference_fc_9_layer_call_and_return_conditional_losses_6305179\HI/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ y
&__inference_fc_9_layer_call_fn_6305188OHI/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ ©
D__inference_flatten_layer_call_and_return_conditional_losses_6305003a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€Р
Ъ Б
)__inference_flatten_layer_call_fn_6305008T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€Р»
B__inference_mnist_layer_call_and_return_conditional_losses_6304354Б$%*+0167<=BCHINOTUZ[>Ґ;
4Ґ1
'К$
input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ »
B__inference_mnist_layer_call_and_return_conditional_losses_6304419Б$%*+0167<=BCHINOTUZ[>Ґ;
4Ґ1
'К$
input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ …
B__inference_mnist_layer_call_and_return_conditional_losses_6304801В$%*+0167<=BCHINOTUZ[?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ …
B__inference_mnist_layer_call_and_return_conditional_losses_6304891В$%*+0167<=BCHINOTUZ[?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ Я
'__inference_mnist_layer_call_fn_6304538t$%*+0167<=BCHINOTUZ[>Ґ;
4Ґ1
'К$
input€€€€€€€€€
p

 
™ "К€€€€€€€€€
Я
'__inference_mnist_layer_call_fn_6304656t$%*+0167<=BCHINOTUZ[>Ґ;
4Ґ1
'К$
input€€€€€€€€€
p 

 
™ "К€€€€€€€€€
†
'__inference_mnist_layer_call_fn_6304944u$%*+0167<=BCHINOTUZ[?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€
†
'__inference_mnist_layer_call_fn_6304997u$%*+0167<=BCHINOTUZ[?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€
£
C__inference_output_layer_call_and_return_conditional_losses_6305239\Z[/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€

Ъ {
(__inference_output_layer_call_fn_6305248OZ[/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€
ґ
%__inference_signature_wrapper_6304711М$%*+0167<=BCHINOTUZ[?Ґ<
Ґ 
5™2
0
input'К$
input€€€€€€€€€"/™,
*
output К
output€€€€€€€€€
