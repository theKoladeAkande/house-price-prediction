import typing as t
import json
from marshmallow import Schema, fields
from marshmallow import ValidationError

from api import config


class InvalidInputError(Exception):
    """Invalid model input."""


SYNTAX_ERROR_FIELD_MAP = {
    '1stFlrSF': 'FirstFlrSF',
    '2ndFlrSF': 'SecondFlrSF',
    '3SsnPorch': 'ThreeSsnPortch'
}


class HouseDataRequestSchema(Schema):
    Alley = fields.Str(allow_none=True)
    BedroomAbvGr = fields.Integer()
    BldgType = fields.Str()
    BsmtCond = fields.Str()
    BsmtExposure = fields.Str(allow_none=True)
    BsmtFinSF1 = fields.Float()
    BsmtFinSF2 = fields.Float()
    BsmtFinType1 = fields.Str()
    BsmtFinType2 = fields.Str()
    BsmtFullBath = fields.Float()
    BsmtHalfBath = fields.Float()
    BsmtQual = fields.Str(allow_none=True)
    BsmtUnfSF = fields.Float()
    CentralAir = fields.Str()
    Condition1 = fields.Str()
    Condition2 = fields.Str()
    Electrical = fields.Str()
    EnclosedPorch = fields.Integer()
    ExterCond = fields.Str()
    ExterQual = fields.Str()
    Exterior1st = fields.Str()
    Exterior2nd = fields.Str()
    Fence = fields.Str(allow_none=True)
    FireplaceQu = fields.Str(allow_none=True)
    Fireplaces = fields.Integer()
    Foundation = fields.Str()
    FullBath = fields.Integer()
    Functional = fields.Str()
    GarageArea = fields.Float()
    GarageCars = fields.Float()
    GarageCond = fields.Str()
    GarageFinish = fields.Str(allow_none=True)
    GarageQual = fields.Str()
    GarageType = fields.Str(allow_none=True)
    GarageYrBlt = fields.Float()
    GrLivArea = fields.Integer()
    HalfBath = fields.Integer()
    Heating = fields.Str()
    HeatingQC = fields.Str()
    HouseStyle = fields.Str()
    Id = fields.Integer()
    KitchenAbvGr = fields.Integer()
    KitchenQual = fields.Str()
    LandContour = fields.Str()
    LandSlope = fields.Str()
    LotArea = fields.Integer()
    LotConfig = fields.Str()
    LotFrontage = fields.Float(allow_none=True)
    LotShape = fields.Str()
    LowQualFinSF = fields.Integer()
    MSSubClass = fields.Integer()
    MSZoning = fields.Str()
    MasVnrArea = fields.Float()
    MasVnrType = fields.Str(allow_none=True)
    MiscFeature = fields.Str(allow_none=True)
    MiscVal = fields.Integer()
    MoSold = fields.Integer()
    Neighborhood = fields.Str()
    OpenPorchSF = fields.Integer()
    OverallCond = fields.Integer()
    OverallQual = fields.Integer()
    PavedDrive = fields.Str()
    PoolArea = fields.Integer()
    PoolQC = fields.Str(allow_none=True)
    RoofMatl = fields.Str()
    RoofStyle = fields.Str()
    SaleCondition = fields.Str()
    SaleType = fields.Str()
    ScreenPorch = fields.Integer()
    Street = fields.Str()
    TotRmsAbvGrd = fields.Integer()
    TotalBsmtSF = fields.Float()
    Utilities = fields.Str()
    WoodDeckSF = fields.Integer()
    YearBuilt = fields.Integer()
    YearRemodAdd = fields.Integer()
    YrSold = fields.Integer()
    FirstFlrSF = fields.Integer()
    SecondFlrSF = fields.Integer()
    ThreeSsnPortch = fields.Integer()


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    schema = HouseDataRequestSchema(strict=True, many=True)
    input_data = json.loads(input_data)
    # convert syntax error field names (beginning with numbers)

    for dict_ in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict_[value] = dict_[key]
            del dict_[key]

    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    
    for dict_ in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict_[key] = dict_[value]
            del dict_[value]

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors