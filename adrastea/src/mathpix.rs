use alloc::collections::BTreeMap;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

pub struct MathPixClient {
    app_id: String,
    app_key: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct DataOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_svg: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_table_html: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_latex: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_tsv: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_asciimath: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_mathml: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ImageToTextOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formats: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_detected_alphabets: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_options: Option<DataOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alphabets_allowed: Option<DetectedAlphabet>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<Rectangle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_blue_hsv_filter: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_line_data: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_word_data: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_smiles: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_geometry_data: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_rotate_confidence_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rm_spaces: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rm_fonts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idiomatic_eqn_arrays: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idiomatic_braces: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numbers_default_to_math: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub math_inline_delimiters: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_spell_check: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_tables_fallback: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ImageToTextResponse {
    pub request_id: Option<String>,
    pub text: Option<String>,
    pub latex_styled: Option<String>,
    pub confidence: Option<f32>,
    pub confidence_rate: Option<f32>,
    pub line_data: Option<Vec<LineData>>,
    pub word_data: Option<Vec<WordData>>,
    pub data: Option<Vec<Data>>,
    pub html: Option<String>,
    pub detected_alphabets: Option<DetectedAlphabet>,
    pub is_printed: Option<bool>,
    pub is_handwritten: Option<bool>,
    pub auto_rotate_confidence: Option<f32>,
    pub geometry_data: Option<Vec<GeometryData>>,
    pub auto_rotate_degrees: Option<i32>,
    pub error: Option<String>,
    pub error_info: Option<BTreeMap<String, serde_json::Value>>,
    pub version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct LineData {
    pub r#type: String,
    pub subtype: Option<String>,
    pub cnt: Vec<(i32, i32)>,
    pub included: bool,
    pub error_id: Option<String>,
    pub text: Option<String>,
    pub confidence: Option<f32>,
    pub confidence_rate: Option<f32>,
    pub after_hyphen: Option<bool>,
    pub html: Option<String>,
    pub data: Option<Vec<Data>>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Data {
    pub r#type: String,
    pub value: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct WordData {
    pub r#type: String,
    pub subtype: Option<String>,
    pub cnt: Vec<(i32, i32)>,
    pub text: Option<String>,
    pub latex: Option<String>,
    pub confidence: Option<f32>,
    pub confidence_rate: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct GeometryData {
    pub position: Option<Rectangle>,
    pub shape_list: Vec<ShapeData>,
    pub label_list: Vec<LabelData>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Rectangle {
    pub top_left_x: i32,
    pub top_left_y: i32,
    pub height: i32,
    pub width: i32,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ShapeData {
    pub r#type: String,
    pub vertex_list: Vec<VertexData>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct VertexData {
    pub x: i32,
    pub y: i32,
    pub edge_list: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct LabelData {
    pub position: Rectangle,
    pub text: String,
    pub latex: String,
    pub confidence: Option<f32>,
    pub confidence_rate: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct DetectedAlphabet {
    pub en: bool,
    pub hi: bool,
    pub zh: bool,
    pub ja: bool,
    pub ko: bool,
    pub ru: bool,
    pub th: bool,
    pub ta: bool,
    pub te: bool,
    pub gu: bool,
    pub bn: bool,
    pub vi: bool,
}

impl MathPixClient {
    pub fn new(app_id: &str, app_key: &str) -> Self {
        Self { app_id: app_id.into(), app_key: app_key.into() }
    }

    pub async fn image_to_text(
        &self, png_bytes: Vec<u8>, options: ImageToTextOptions,
    ) -> anyhow::Result<ImageToTextResponse> {
        let url = "https://api.mathpix.com/v3/text";
        let client = reqwest::Client::new();
        let json = serde_json::to_string(&options)?;
        let form = Form::new()
            .text("options_json", json)
            .part("file", Part::bytes(png_bytes).mime_str("image/png")?.file_name("image.png"));
        let response = client
            .post(url)
            .header("app_id", &self.app_id)
            .header("app_key", &self.app_key)
            .multipart(form)
            .send()
            .await?;
        let response = response.text().await?;
        Ok(serde_json::from_str(&response)?)
    }
}
