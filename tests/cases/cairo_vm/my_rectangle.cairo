use debug::PrintTrait;

#[derive(Copy, Drop)]
struct Rectangle{
height: u64,
width: u64,
}

impl RectangleImplTrait of PrintTrait<Rectangle> {
fn print(self: Rectangle){
self.height.print();
self.width.print();
}
}

trait RectangleTrait {
fn my_perimeter (self: @Rectangle) -> u64;
}

impl RectangleTraitImpl of RectangleTrait{
fn my_perimeter(self: @Rectangle)-> u64{
*self.height + 2 + *self.width*2
}}


fn main(){
let rectangle = Rectangle{
height: 2,
width: 2
};

// rectangle.print();
rectangle.my_perimeter().print();

}