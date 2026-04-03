"""
============================================================
Driver Drowsiness Detection — Training Script
============================================================
Student : Hrishitha Prasad A S
USN     : 1NT23AD022

Dataset : Drowsy_datset (Yashar Jebraeily)
Classes : DROWSY / NATURAL
Models  : VGG16, MobileNetV2, ResNet50V2, DrowsyNet

Run     : python train.py
Output  : models/ folder with 4 .keras files
============================================================
"""

import os, random, warnings, json, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D,
    Reshape, Bidirectional, LSTM, Multiply,
    Input, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50V2
from tensorflow.keras.applications.vgg16        import preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.resnet_v2    import preprocess_input as res_pre
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

print("\n" + "="*60)
print("  DRIVER DROWSINESS DETECTION — TRAINING")
print("="*60)
print(f"  TensorFlow version : {tf.__version__}")
print(f"  GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")
print("="*60 + "\n")

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR   = os.path.join(DATASET_DIR, 'train')
TEST_DIR    = os.path.join(DATASET_DIR, 'test')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Verify dataset
assert os.path.exists(TRAIN_DIR), f"❌ Train folder not found: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR),  f"❌ Test folder not found: {TEST_DIR}"

# Count images
for split, path in [('Train', TRAIN_DIR), ('Test', TEST_DIR)]:
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
            print(f"  {split:6s} / {cls:10s} → {count} images")

print()

# ── Settings ─────────────────────────────────────────────────────
IMG_TRANSFER = (224, 224)
IMG_CUSTOM   = (64, 64)
BATCH        = 32
EPOCHS       = 3
SEED         = 42
VAL_SPLIT    = 0.15

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ALL_RESULTS  = {}
PARAM_COUNTS = {}

# ── Data generators ──────────────────────────────────────────────
print("🔄 Setting up data generators...")

# Transfer learning generators (224x224 RGB)
def make_transfer_gens(preprocess_fn):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=[0.6, 1.4],
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=VAL_SPLIT
    )
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_TRANSFER,
        color_mode='rgb',
        batch_size=BATCH,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=SEED
    )
    val = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_TRANSFER,
        color_mode='rgb',
        batch_size=BATCH,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED
    )
    test = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_TRANSFER,
        color_mode='rgb',
        batch_size=BATCH,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )
    return train, val, test

# Custom generator (64x64 grayscale) for DrowsyNet
def make_custom_gens():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=[0.6, 1.4],
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest',
        validation_split=VAL_SPLIT
    )
    test_gen = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_CUSTOM,
        color_mode='grayscale',
        batch_size=BATCH,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=SEED
    )
    val = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_CUSTOM,
        color_mode='grayscale',
        batch_size=BATCH,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED
    )
    test = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_CUSTOM,
        color_mode='grayscale',
        batch_size=BATCH,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )
    return train, val, test

# Make base generators to get class info and weights
base_train, base_val, base_test = make_transfer_gens(vgg_pre)

CLASS_INDICES = base_train.class_indices
print(f"\n  Class indices : {CLASS_INDICES}")
print(f"  (0 = {[k for k,v in CLASS_INDICES.items() if v==0][0]}, "
      f"1 = {[k for k,v in CLASS_INDICES.items() if v==1][0]})")

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(base_train.classes),
    y=base_train.classes
)
CLASS_WEIGHT_DICT = dict(enumerate(class_weights))
print(f"  Class weights : {CLASS_WEIGHT_DICT}")
print(f"\n  Train samples : {base_train.samples}")
print(f"  Val samples   : {base_val.samples}")
print(f"  Test samples  : {base_test.samples}")

# ── Helper functions ─────────────────────────────────────────────

def plot_history(hist, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'{model_name} — Training History', fontsize=13, fontweight='bold')

    axes[0].plot(hist.history['accuracy'],     label='Train', color='royalblue')
    axes[0].plot(hist.history['val_accuracy'], label='Val',   color='tomato')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist.history['loss'],     label='Train', color='royalblue')
    axes[1].plot(hist.history['val_loss'], label='Val',   color='tomato')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{model_name}_history.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  📊 History plot saved → results/{model_name}_history.png")


def evaluate_model(model, test_gen, model_name):
    test_gen.reset()
    preds_raw = model.predict(test_gen, verbose=0)
    preds     = (preds_raw > 0.5).astype(int).flatten()
    labels    = test_gen.classes[:len(preds)]

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, preds_raw.flatten())
    except Exception:
        auc = 0.0

    print(f"\n  ── {model_name} Results ──────────────────")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print()
    print(classification_report(
        labels, preds,
        target_names=list(CLASS_INDICES.keys())
    ))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_INDICES.keys()),
                yticklabels=list(CLASS_INDICES.keys()))
    plt.title(f'{model_name} — Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{model_name}_confusion.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()

    ALL_RESULTS[model_name] = {
        'accuracy' : round(acc,  4),
        'precision': round(prec, 4),
        'recall'   : round(rec,  4),
        'f1'       : round(f1,   4),
        'auc'      : round(auc,  4),
    }
    return acc


def get_callbacks(model_name, monitor='val_accuracy'):
    ckpt_path = os.path.join(MODELS_DIR, f'{model_name}_best.keras')
    return [
        EarlyStopping(
            monitor=monitor, patience=10,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            ckpt_path, monitor=monitor,
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-7, verbose=1
        )
    ]


# ════════════════════════════════════════════════════════════════
# MODEL 1 — VGG16
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 1/4 — VGG16")
print("="*60)

train_v, val_v, test_v = make_transfer_gens(vgg_pre)

base_vgg = VGG16(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
for layer in base_vgg.layers[:-8]:
    layer.trainable = False
for layer in base_vgg.layers[-8:]:
    layer.trainable = True

vgg_model = Sequential([
    base_vgg,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name='VGG16_Drowsiness')

vgg_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
PARAM_COUNTS['VGG16'] = vgg_model.count_params()
print(f"  Parameters: {vgg_model.count_params():,}")

print("\n🚀 Training VGG16...")
t0 = time.time()
hist_v = vgg_model.fit(
    train_v,
    validation_data=val_v,
    epochs=EPOCHS,
    callbacks=get_callbacks('VGG16'),
    class_weight=CLASS_WEIGHT_DICT,
    verbose=1
)
print(f"  ⏱  Training time: {(time.time()-t0)/60:.1f} min")

plot_history(hist_v, 'VGG16')
evaluate_model(vgg_model, test_v, 'VGG16')
vgg_model.save(os.path.join(MODELS_DIR, 'vgg16_final.keras'))
print("  ✅ VGG16 saved → models/vgg16_final.keras")


# ════════════════════════════════════════════════════════════════
# MODEL 2 — MobileNetV2
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 2/4 — MobileNetV2")
print("="*60)

train_m, val_m, test_m = make_transfer_gens(mob_pre)

base_mob = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
for layer in base_mob.layers[:-20]:
    layer.trainable = False
for layer in base_mob.layers[-20:]:
    layer.trainable = True

mob_model = Sequential([
    base_mob,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name='MobileNetV2_Drowsiness')

mob_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
PARAM_COUNTS['MobileNetV2'] = mob_model.count_params()
print(f"  Parameters: {mob_model.count_params():,}")

print("\n🚀 Training MobileNetV2...")
t0 = time.time()
hist_m = mob_model.fit(
    train_m,
    validation_data=val_m,
    epochs=EPOCHS,
    callbacks=get_callbacks('MobileNetV2'),
    class_weight=CLASS_WEIGHT_DICT,
    verbose=1
)
print(f"  ⏱  Training time: {(time.time()-t0)/60:.1f} min")

plot_history(hist_m, 'MobileNetV2')
evaluate_model(mob_model, test_m, 'MobileNetV2')
mob_model.save(os.path.join(MODELS_DIR, 'mobilenet_final.keras'))
print("  ✅ MobileNetV2 saved → models/mobilenet_final.keras")


# ════════════════════════════════════════════════════════════════
# MODEL 3 — ResNet50V2
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 3/4 — ResNet50V2")
print("="*60)

train_r, val_r, test_r = make_transfer_gens(res_pre)

base_res = ResNet50V2(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
for layer in base_res.layers[:-20]:
    layer.trainable = False
for layer in base_res.layers[-20:]:
    layer.trainable = True

res_model = Sequential([
    base_res,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name='ResNet50V2_Drowsiness')

res_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
PARAM_COUNTS['ResNet50V2'] = res_model.count_params()
print(f"  Parameters: {res_model.count_params():,}")

print("\n🚀 Training ResNet50V2...")
t0 = time.time()
hist_r = res_model.fit(
    train_r,
    validation_data=val_r,
    epochs=EPOCHS,
    callbacks=get_callbacks('ResNet50V2'),
    class_weight=CLASS_WEIGHT_DICT,
    verbose=1
)
print(f"  ⏱  Training time: {(time.time()-t0)/60:.1f} min")

plot_history(hist_r, 'ResNet50V2')
evaluate_model(res_model, test_r, 'ResNet50V2')
res_model.save(os.path.join(MODELS_DIR, 'resnet_final.keras'))
print("  ✅ ResNet50V2 saved → models/resnet_final.keras")


# ════════════════════════════════════════════════════════════════
# MODEL 4 — DrowsyNet (Custom CNN + SE Attention + BiLSTM)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL 4/4 — DrowsyNet (Custom)")
print("="*60)

train_c, val_c, test_c = make_custom_gens()

# Recompute class weights for custom generators
cw_custom = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_c.classes),
    y=train_c.classes
)
cw_custom_dict = dict(enumerate(cw_custom))

def se_block(x, ratio=16):
    ch = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(1, ch // ratio), activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, 1, ch))(se)
    return Multiply()([x, se])

def build_drowsynet(shape=(64, 64, 1)):
    inp = Input(shape=shape)

    x = Conv2D(32, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = se_block(x, ratio=16)

    sh = x.shape
    x  = Reshape((sh[1], sh[2] * sh[3]))(x)
    x  = Bidirectional(LSTM(64, return_sequences=False))(x)

    x   = Dropout(0.5)(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inputs=inp, outputs=out, name='DrowsyNet')

drowsynet = build_drowsynet()
drowsynet.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
PARAM_COUNTS['DrowsyNet'] = drowsynet.count_params()
drowsynet.summary()
print(f"\n  Parameters: {drowsynet.count_params():,}")

print("\n🚀 Training DrowsyNet...")
t0 = time.time()
hist_d = drowsynet.fit(
    train_c,
    validation_data=val_c,
    epochs=3,
    callbacks=get_callbacks('DrowsyNet'),
    class_weight=cw_custom_dict,
    verbose=1
)
print(f"  ⏱  Training time: {(time.time()-t0)/60:.1f} min")

plot_history(hist_d, 'DrowsyNet')
evaluate_model(drowsynet, test_c, 'DrowsyNet')
drowsynet.save(os.path.join(MODELS_DIR, 'drowsynet_final.keras'))
print("  ✅ DrowsyNet saved → models/drowsynet_final.keras")


# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(f"  {'Model':<15} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print("  " + "-"*45)
for name, r in ALL_RESULTS.items():
    print(f"  {name:<15} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")

best = max(ALL_RESULTS, key=lambda m: ALL_RESULTS[m]['accuracy'])
print(f"\n  🏆 Best Model: {best} ({ALL_RESULTS[best]['accuracy']*100:.2f}% accuracy)")

# Save class info for app.py
class_info = {
    "class_indices"  : CLASS_INDICES,
    "index_to_label" : {str(v): k for k, v in CLASS_INDICES.items()},
    "best_model"     : best,
    "results"        : ALL_RESULTS,
    "param_counts"   : PARAM_COUNTS,
}
with open(os.path.join(MODELS_DIR, 'class_info.json'), 'w') as f:
    json.dump(class_info, f, indent=2)

print("\n  ✅ class_info.json saved → models/class_info.json")
print("\n" + "="*60)
print("  ALL MODELS SAVED TO: models/")
print("  - vgg16_final.keras")
print("  - mobilenet_final.keras")
print("  - resnet_final.keras")
print("  - drowsynet_final.keras")
print("  - class_info.json")
print("="*60)
print("\n  ✅ TRAINING COMPLETE! Now copy the models/ folder")
print("  into your app folder and run python app.py\n")
